import unittest
import numpy as np
import cv2
import onnxruntime as ort
from inference import MODEL, Predictor, gradcam_map, vessel_candidates

class ExplanationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Predictor()
        cls.image = np.random.default_rng(7).integers(0,256,(224,224,3),dtype=np.uint8)
    def test_unchanged_scores(self):
        original = ort.InferenceSession(str(MODEL), providers=['CPUExecutionProvider'])
        expected = original.run(None, {'data': self.image.transpose(2,0,1)[None].astype(np.float32)})[0][0]
        np.testing.assert_allclose(self.engine.scores(self.image), expected, atol=1e-6,rtol=1e-5)
    def test_gradient_finite_difference(self):
        _, features = self.engine.forward(self.image)
        a = features.astype(np.float64)
        weights = self.engine.weights.astype(np.float64)
        for cls in range(5):
            k = int(np.argmax(np.abs(weights[:,cls])))
            changed = a.copy(); changed[k,2,3] += .001
            difference = ((changed.mean((1,2)) @ weights)[cls] - (a.mean((1,2)) @ weights)[cls])/.001
            self.assertAlmostEqual(difference,weights[k,cls]/49,places=8)
    def test_maps_class_specific_and_zero(self):
        _, features = self.engine.forward(self.image)
        maps = [gradcam_map(features,self.engine.weights,c) for c in range(5)]
        self.assertTrue(any(not np.allclose(maps[0],m) for m in maps[1:]))
        self.assertTrue(all(m.shape==(224,224) and np.isfinite(m).all() and m.min()>=0 and m.max()<=1 for m in maps))
        self.assertEqual(gradcam_map(np.zeros_like(features),self.engine.weights,0).max(),0)
    def test_vessels_detect_dark_line_and_blank(self):
        uniform=np.full((224,224,3),160,np.uint8)
        self.assertEqual(vessel_candidates(uniform).max(),0)
        uniform[40:185,110:113]=30
        mask=vessel_candidates(uniform)
        self.assertGreater((mask[45:180,110:113]>0).mean(),.9)
        self.assertEqual(mask[50:170,30:60].max(),0)
        self.assertTrue(set(np.unique(mask)) <= {0,255})
if __name__=='__main__':unittest.main()
