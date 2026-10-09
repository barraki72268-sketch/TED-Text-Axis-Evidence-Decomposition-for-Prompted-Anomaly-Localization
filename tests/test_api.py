import base64
from concurrent.futures import ThreadPoolExecutor
import io
import threading
import unittest

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from ted.inference.api import create_app


def png():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeEngine:
    def info(self):
        return {"host": "test-only", "artifact_sha256": "a" * 64}

    def predict(self, image):
        return dict(host_map=np.zeros((1, 8, 8)), cted_map=np.ones((1, 8, 8)),
                    image_score=0.4, artifact_sha256="a" * 64, timing_ms={"host": 1.0})


class APIContractTests(unittest.TestCase):
    def test_only_recorded_strengths_are_dispatched_and_preserved(self):
        seen = []
        class AlphaEngine(FakeEngine):
            def info(self):
                return dict(super().info(), alphas=[.5, 1.5])
            def predict_for_alpha(self, image, alpha):
                seen.append(alpha)
                return dict(super().predict(image), alpha=alpha)
        with TestClient(create_app(AlphaEngine)) as client:
            for alpha in [.5, 1.5]:
                response = client.post('/predict?alpha=' + str(alpha), content=png(),
                                       headers={'Content-Type': 'image/png'})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['alpha'], alpha)
            for query in ['alpha=.7', 'alpha=nan', 'alpha=inf', 'alpha=.5&category=01']:
                response = client.post('/predict?' + query, content=png(),
                                       headers={'Content-Type': 'image/png'})
                self.assertEqual(response.status_code, 422)
        self.assertEqual(seen, [.5, 1.5])
        with TestClient(create_app(FakeEngine)) as client:
            response = client.post('/predict?alpha=.5', content=png(),
                                   headers={'Content-Type': 'image/png'})
            self.assertEqual(response.status_code, 422)

    def test_corrected_score_is_preserved_and_nonfinite_score_is_rejected(self):
        class CorrectedEngine(FakeEngine):
            def predict(self, image):
                return dict(super().predict(image), cted_image_score=0.75)
        with TestClient(create_app(CorrectedEngine)) as client:
            result = client.post('/predict', content=png(), headers={'Content-Type':'image/png'}).json()
            self.assertEqual(result['image_score'], 0.4)
            self.assertEqual(result['cted_image_score'], 0.75)
        class NonfiniteEngine(FakeEngine):
            def predict(self, image):
                return dict(super().predict(image), cted_image_score=float('nan'))
        from ted.inference.api import encode_prediction
        with self.assertRaisesRegex(RuntimeError, 'corrected image score'):
            encode_prediction(NonfiniteEngine(), png(), 25_000_000)

    def test_category_selection_preserves_host_specific_score_policy(self):
        class CategoryEngine(FakeEngine):
            def info(self):
                return dict(super().info(), categories=['01', '02'], category=None)
            def predict_for_category(self, image, category):
                return dict(super().predict(image), category=category,
                            image_score_policy='raw detection score; not dataset-normalized paper metric')
        with TestClient(create_app(CategoryEngine)) as client:
            self.assertEqual(client.post('/predict', content=png(), headers={'Content-Type':'image/png'}).status_code, 422)
            self.assertEqual(client.post('/predict?category=invalid', content=png(), headers={'Content-Type':'image/png'}).status_code, 422)
            response = client.post('/predict?category=02', content=png(), headers={'Content-Type':'image/png'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['category'], '02')
            self.assertIn('not dataset-normalized', response.json()['image_score_policy'])
        with TestClient(create_app(FakeEngine)) as client:
            self.assertEqual(client.post('/predict?category=02', content=png(), headers={'Content-Type':'image/png'}).status_code, 422)

    def test_lifecycle_and_outputs(self):
        calls = []
        def factory():
            calls.append(1)
            return FakeEngine()
        app = create_app(factory)
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/ready").status_code, 200)
            self.assertEqual(client.get("/model-info").json()["host"], "test-only")
            for _ in range(2):
                response = client.post("/predict", content=png(), headers={"Content-Type": "image/png"})
                self.assertEqual(response.status_code, 200)
                data = response.json()
                with np.load(io.BytesIO(base64.b64decode(data["maps_npz_base64"])), allow_pickle=False) as maps:
                    np.testing.assert_array_equal(maps["cted"], np.ones((1, 8, 8)))
                self.assertEqual(data["display"]["shared_range"], [0.0, 1.0])
                self.assertNotIn("is_anomaly", data)
            self.assertEqual(len(calls), 1)
        self.assertIsNone(app.state.engine)

    def test_bad_input_and_limits(self):
        with TestClient(create_app(FakeEngine)) as client:
            self.assertEqual(client.post("/predict", content=png()).status_code, 415)
            self.assertEqual(client.post("/predict", content=b"not PNG", headers={"Content-Type": "image/png"}).status_code, 400)
        for options in ({"max_body_bytes": 8}, {"max_pixels": 4}):
            with TestClient(create_app(FakeEngine, **options)) as client:
                self.assertEqual(client.post("/predict", content=png(), headers={"Content-Type": "image/png"}).status_code, 413)

    def test_busy_rejection(self):
        entered, release = threading.Event(), threading.Event()
        class SlowEngine(FakeEngine):
            def predict(self, image):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("Test timed out")
                return super().predict(image)
        with TestClient(create_app(SlowEngine)) as client, ThreadPoolExecutor(1) as pool:
            first = pool.submit(client.post, "/predict", content=png(), headers={"Content-Type": "image/png"})
            try:
                self.assertTrue(entered.wait(3))
                busy = client.post("/predict", content=png(), headers={"Content-Type": "image/png"})
                self.assertEqual(busy.status_code, 503)
                self.assertEqual(busy.headers["retry-after"], "1")
                self.assertEqual(client.get("/health").status_code, 200)
            finally:
                release.set()
            self.assertEqual(first.result().status_code, 200)

    def test_failure_does_not_disclose_paths(self):
        class BrokenEngine(FakeEngine):
            def predict(self, image):
                raise RuntimeError("private/checkpoint/path")
        with TestClient(create_app(BrokenEngine)) as client:
            response = client.post("/predict", content=png(), headers={"Content-Type": "image/png"})
            self.assertEqual(response.status_code, 500)
            self.assertNotIn("private", response.text)


if __name__ == "__main__":
    unittest.main()
