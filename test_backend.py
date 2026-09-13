import io
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import server

def encode(a):
    output=io.BytesIO(); Image.fromarray(a).save(output,format='PNG');return output.getvalue()

class BackendChecks(unittest.TestCase):
    def setUp(self):
        self.client=server.app.test_client()
    def post(self, data, origin='http://localhost'):
        return self.client.post('/api/screen', data=data, content_type='application/octet-stream',
                                headers={'Origin':origin})
    def test_model_and_frontend(self):
        self.assertEqual(self.client.get('/healthz').status_code,200)
        self.assertTrue(self.client.get('/api/status').json['model'])
        self.assertIn(b'Migration test build',self.client.get('/').data)
    def test_black_image_rejected_without_grade(self):
        response=self.post(encode(np.zeros((224,224,3),np.uint8)))
        self.assertEqual(response.status_code,200)
        self.assertFalse(response.json['quality']['accepted'])
        self.assertIsNone(response.json['grade'])
    def test_synthetic_execution_only(self):
        # Deliberately NOT a fundus image; proves execution, not medical validity.
        rgb=np.random.default_rng(3).integers(0,256,(300,300,3),dtype=np.uint8)
        response=self.post(encode(rgb))
        self.assertEqual(response.status_code,200)
        self.assertIn(response.json['grade'],range(5))
        self.assertTrue(response.json['heatmap'].startswith('data:image/png;base64,'))
        self.assertTrue(response.json['vessels'].startswith('data:image/png;base64,'))
    def test_input_and_origin(self):
        self.assertEqual(self.post(b'bad').status_code,400)
        self.assertEqual(self.post(b'bad','https://other.example').status_code,403)
        self.assertEqual(self.post(encode(np.zeros((100,100,3),np.uint8))).status_code,400)
        self.assertEqual(self.post(b'x'*(15*1024*1024+1)).status_code,413)
    def test_busy(self):
        with server.LOCK:
            self.assertEqual(self.post(b'x').status_code,429)
    def test_auth(self):
        import base64
        with patch.object(server,'PASSWORD','sample-test-password'):
            self.assertEqual(self.client.get('/').status_code,401)
            token=base64.b64encode(b'demo:sample-test-password').decode()
            self.assertEqual(self.client.get('/',headers={'Authorization':'Basic '+token}).status_code,200)

if __name__=='__main__':unittest.main()
