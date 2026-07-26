"""
AI Pro - YOLOv8 Disease Detector
Handles model loading, inference, and result processing.
Supports BOTH detection models (bounding boxes) and classification models
(whole-image single label, e.g. a model trained on PlantVillage) — the
correct handling is chosen automatically based on the loaded model's task.

Supports up to THREE models loaded alongside each other (model_path,
model_path_v2, model_path_v3) — e.g. your original crop model plus two
newly trained models covering additional crops. Results from all loaded
models are merged into a single response so the rest of the app doesn't
need to change.
"""
import os
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from pathlib import Path
from plantvillage_classes import get_plantvillage_info

class CropDiseaseDetector:
    """YOLOv8-based crop disease detection/classification engine"""

    def __init__(self, model_path=None, conf_threshold=0.25, iou_threshold=0.45,
                 model_path_v2=None, model_path_v3=None):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.class_names = {}
        self.model_loaded = False
        self.model_task = None  # 'detect' or 'classify'

        # Secondary model (e.g. new crops added later)
        self.model_v2 = None
        self.class_names_v2 = {}
        self.model_v2_loaded = False
        self.model_v2_task = None

        # Tertiary model (e.g. a further round of new crops/classes)
        self.model_v3 = None
        self.class_names_v3 = {}
        self.model_v3_loaded = False
        self.model_v3_task = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"⚠️ Model not found at {model_path}. Using fallback detection.")

        if model_path_v2 and os.path.exists(model_path_v2):
            self.load_model_v2(model_path_v2)
        elif model_path_v2:
            print(f"⚠️ Secondary model not found at {model_path_v2}. Skipping.")

        if model_path_v3 and os.path.exists(model_path_v3):
            self.load_model_v3(model_path_v3)
        elif model_path_v3:
            print(f"⚠️ Tertiary model not found at {model_path_v3}. Skipping.")

    def load_model(self, model_path):
        """Load YOLOv8 model (detection or classification)"""
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            self.model_task = getattr(self.model, 'task', 'detect')
            self.model_loaded = True
            print(f"✅ YOLOv8 model loaded on {self.device} (task: {self.model_task})")
            print(f"   Classes: {len(self.class_names)}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model_loaded = False
            return False

    def load_model_v2(self, model_path):
        """Load a second YOLOv8 model (e.g. covers additional crops)"""
        try:
            self.model_v2 = YOLO(model_path)
            self.class_names_v2 = self.model_v2.names
            self.model_v2_task = getattr(self.model_v2, 'task', 'detect')
            self.model_v2_loaded = True
            print(f"✅ Secondary YOLOv8 model loaded on {self.device} (task: {self.model_v2_task})")
            print(f"   Classes: {len(self.class_names_v2)}")
            return True
        except Exception as e:
            print(f"❌ Error loading secondary model: {e}")
            self.model_v2_loaded = False
            return False

    def load_model_v3(self, model_path):
        """Load a third YOLOv8 model (e.g. a further round of new crops)"""
        try:
            self.model_v3 = YOLO(model_path)
            self.class_names_v3 = self.model_v3.names
            self.model_v3_task = getattr(self.model_v3, 'task', 'detect')
            self.model_v3_loaded = True
            print(f"✅ Tertiary YOLOv8 model loaded on {self.device} (task: {self.model_v3_task})")
            print(f"   Classes: {len(self.class_names_v3)}")
            return True
        except Exception as e:
            print(f"❌ Error loading tertiary model: {e}")
            self.model_v3_loaded = False
            return False

    def detect(self, image_path, crop_group='auto'):
        """Run inference on image and return structured, merged results.

        crop_group controls which model(s) run:
          'auto' (default) - run every loaded model and merge results
          'v1'  - only the primary model (e.g. PlantVillage crops)
          'v2'  - only the secondary model
          'v3'  - only the tertiary model
        This avoids a specialist model being forced to guess on a crop
        it was never trained on, which otherwise produces confident-looking
        but meaningless results.
        """
        if not self.model_loaded and not self.model_v2_loaded and not self.model_v3_loaded:
            return self._fallback_detection(image_path)

        run_v1 = self.model_loaded and crop_group in ('auto', 'v1')
        run_v2 = self.model_v2_loaded and crop_group in ('auto', 'v2')
        run_v3 = self.model_v3_loaded and crop_group in ('auto', 'v3')

        results_list = []

        if run_v1:
            r = self._run_single_model(
                self.model, self.model_task, self.class_names, image_path,
                suffix='v1', use_plantvillage_lookup=True
            )
            if r:
                results_list.append(r)

        if run_v2:
            r = self._run_single_model(
                self.model_v2, self.model_v2_task, self.class_names_v2, image_path,
                suffix='v2', use_plantvillage_lookup=False
            )
            if r:
                results_list.append(r)

        if run_v3:
            r = self._run_single_model(
                self.model_v3, self.model_v3_task, self.class_names_v3, image_path,
                suffix='v3', use_plantvillage_lookup=False
            )
            if r:
                results_list.append(r)

        if not results_list:
            return self._fallback_detection(image_path)

        return self._merge_results(results_list)

    def _run_single_model(self, model, task, class_names, image_path, suffix='v1',
                           use_plantvillage_lookup=False):
        """Run one model (detection or classification) and return its raw
        processed result dict, or None on failure."""
        try:
            if task == 'classify':
                results = model(image_path, device=self.device, verbose=False)
                return self._process_classification_results(
                    results, class_names, use_plantvillage_lookup
                )

            results = model(
                image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            return self._process_results(results, class_names, image_path, suffix)

        except Exception as e:
            print(f"❌ Detection error: {e}")
            return None

    def _merge_results(self, results_list):
        """Combine detections from all loaded models into a single response."""
        all_detections = []
        model_names = []
        annotated_image = None

        for r in results_list:
            if not r:
                continue
            all_detections.extend(r['detections'])
            model_names.append(r['model'])
            if annotated_image is None and r.get('annotated_image'):
                annotated_image = r['annotated_image']

        # Sort combined detections: real bounding-box detections first (these
        # only appear when a model found actual visual evidence), then
        # classification-only guesses (which are forced to answer even with
        # no real match, so their confidence numbers aren't directly
        # comparable to a detection model's). Within each group, sort by
        # confidence, highest first.
        all_detections.sort(
            key=lambda x: (x.get('bbox') is None, -x.get('confidence', 0))
        )

        return {
            'success': True,
            'detections': all_detections,
            'annotated_image': annotated_image,
            'total_detections': len(all_detections),
            'model': ' + '.join(model_names) if model_names else 'Unknown'
        }

    def _process_classification_results(self, results, class_names, use_plantvillage_lookup=False):
        """Process YOLOv8 classification results into the SAME response shape
        as detection results, so the rest of the app (app.py, disease_database
        lookups) works unchanged.

        use_plantvillage_lookup=True is for models trained on PlantVillage-style
        class names (e.g. 'Tomato___Septoria_leaf_spot'). Other classification
        models (e.g. a custom-trained Banana/Chilli/Cauliflower model) use their
        own naming convention, so we parse crop/disease from the class name
        directly instead of forcing a PlantVillage lookup that wouldn't match.
        """
        result = results[0]
        probs = result.probs

        top_idx = int(probs.top1)
        confidence = float(probs.top1conf) * 100
        raw_class_name = class_names.get(top_idx, f"class_{top_idx}")

        if use_plantvillage_lookup:
            info = get_plantvillage_info(raw_class_name)
            crop_key = info['crop_key']
            crop_name = info['crop_name']
            crop_icon = info['crop_icon']
            disease_name = info['disease_name']
            severity = info['severity']
            model_label = 'YOLOv8-cls (PlantVillage)'
        else:
            # Parse "Crop_Disease_Name" style labels directly, similar to
            # how detection results are parsed.
            parts = raw_class_name.replace('_', ' ').replace('-', ' ').split()
            crop_key = parts[0].lower() if parts else 'unknown'
            crop_name = parts[0] if parts else 'Unknown'
            crop_icon = '🌱'
            disease_name = ' '.join(parts[1:]) if len(parts) > 1 else 'Unknown'
            severity = 'Healthy' if 'healthy' in raw_class_name.lower() else 'Unknown'
            model_label = 'YOLOv8-cls'

        detections = [{
            'class_id': top_idx,
            'class_name': raw_class_name,
            'confidence': round(confidence, 1),
            'bbox': None,  # classification has no bounding box
            'crop': crop_key,
            'crop_name': crop_name,
            'crop_icon': crop_icon,
            'disease': disease_name,
            'severity': severity,
        }]

        return {
            'success': True,
            'detections': detections,
            'annotated_image': None,  # no annotated overlay for classification
            'total_detections': 1,
            'model': model_label
        }

    def _process_results(self, results, class_names, image_path, suffix='v1'):
        """Process YOLOv8 detection results into structured format"""
        result = results[0]  # Single image

        # Get detections
        boxes = result.boxes

        detections = []
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names.get(cls_id, f"class_{cls_id}")

                # Parse class name (format: "crop_disease" or "crop disease")
                parts = class_name.replace('_', ' ').replace('-', ' ').split()

                detections.append({
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': round(conf * 100, 1),
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'crop': parts[0] if len(parts) > 0 else 'unknown',
                    'disease': ' '.join(parts[1:]) if len(parts) > 1 else 'unknown'
                })

        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        # Get annotated image
        annotated_img = result.plot()

        # Save annotated image (suffix keeps v1/v2 outputs from overwriting each other)
        output_path = image_path.replace('/uploads/', f'/uploads/annotated_{suffix}_')
        cv2.imwrite(output_path, annotated_img)

        return {
            'success': True,
            'detections': detections,
            'annotated_image': output_path,
            'total_detections': len(detections),
            'model': 'YOLOv8'
        }

    def _fallback_detection(self, image_path):
        """Fallback: Use image analysis when no model is available"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                img = np.array(Image.open(image_path).convert('RGB'))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Simple color-based analysis
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Calculate color statistics
            h_mean = np.mean(hsv[:,:,0])
            s_mean = np.mean(hsv[:,:,1])
            v_mean = np.mean(hsv[:,:,2])

            # Detect spots (brown/dark areas)
            lower_brown = np.array([0, 50, 20])
            upper_brown = np.array([30, 255, 150])
            brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
            brown_ratio = np.sum(brown_mask > 0) / (brown_mask.shape[0] * brown_mask.shape[1])

            # Detect yellow areas
            lower_yellow = np.array([20, 50, 50])
            upper_yellow = np.array([40, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            yellow_ratio = np.sum(yellow_mask > 0) / (yellow_mask.shape[0] * yellow_mask.shape[1])

            # Determine severity and disease based on ratios
            if brown_ratio > 0.15 or yellow_ratio > 0.20:
                severity = 'High'
                confidence = min(85, 50 + int((brown_ratio + yellow_ratio) * 200))
            elif brown_ratio > 0.08 or yellow_ratio > 0.12:
                severity = 'Medium'
                confidence = min(75, 45 + int((brown_ratio + yellow_ratio) * 200))
            elif brown_ratio > 0.03 or yellow_ratio > 0.05:
                severity = 'Low'
                confidence = min(65, 35 + int((brown_ratio + yellow_ratio) * 200))
            else:
                severity = 'Healthy'
                confidence = min(95, 70 + int((1 - brown_ratio - yellow_ratio) * 50))

            # Create pixel grid visualization
            grid_size = 10
            h, w = img.shape[:2]
            patch_h, patch_w = h // grid_size, w // grid_size
            pixel_grid = []

            for i in range(grid_size):
                for j in range(grid_size):
                    patch = img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

                    avg_color = np.mean(patch, axis=(0,1)).astype(int).tolist()

                    # Classify patch
                    patch_brown = cv2.inRange(patch_hsv, lower_brown, upper_brown)
                    patch_yellow = cv2.inRange(patch_hsv, lower_yellow, upper_yellow)

                    brown_pct = np.sum(patch_brown > 0) / (patch_h * patch_w)
                    yellow_pct = np.sum(patch_yellow > 0) / (patch_h * patch_w)

                    if brown_pct > 0.1 or yellow_pct > 0.15:
                        patch_type = 'spot'
                    elif avg_color[1] > avg_color[2] and avg_color[1] > 60:  # More green
                        patch_type = 'healthy'
                    else:
                        patch_type = 'other'

                    pixel_grid.append({
                        'r': avg_color[2], 'g': avg_color[1], 'b': avg_color[0],
                        'type': patch_type
                    })

            # Determine likely crop from dominant color
            crop_guess = self._guess_crop(h_mean, s_mean, v_mean)

            return {
                'success': True,
                'detections': [{
                    'class_name': f'{crop_guess["crop"]}_{severity}',
                    'confidence': confidence,
                    'crop': crop_guess['crop'],
                    'disease': severity if severity != 'Healthy' else 'Healthy',
                    'severity': severity,
                    'bbox': [0, 0, w, h]
                }],
                'annotated_image': image_path,
                'total_detections': 1,
                'model': 'Fallback Color Analysis',
                'pixel_grid': pixel_grid,
                'analysis': {
                    'brown_ratio': round(brown_ratio * 100, 1),
                    'yellow_ratio': round(yellow_ratio * 100, 1),
                    'h_mean': round(h_mean, 1),
                    's_mean': round(s_mean, 1),
                    'v_mean': round(v_mean, 1)
                }
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'detections': [],
                'model': 'Error'
            }

    def _guess_crop(self, h, s, v):
        """Guess crop type from HSV color signature"""
        # Simple heuristic based on hue
        if 35 <= h <= 75 and s > 40:  # Green
            return {'crop': 'tomato', 'icon': '🍅'}
        elif 20 <= h <= 40 and s > 50:  # Yellow-green
            return {'crop': 'corn', 'icon': '🌽'}
        elif h > 75 and s > 30:  # Blue-green
            return {'crop': 'rice', 'icon': '🍚'}
        elif h < 15 and s > 30:  # Reddish
            return {'crop': 'cotton', 'icon': '🧵'}
        else:
            return {'crop': 'unknown', 'icon': '🌱'}

    def train(self, data_yaml, epochs=100, imgsz=640, batch=16, project='runs/train', name='cropguard'):
        """Train YOLOv8 model on custom dataset"""
        if not self.model_loaded:
            print("❌ No model loaded. Cannot train.")
            return None

        try:
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project=project,
                name=name,
                device=self.device,
                patience=20,
                save=True,
                plots=True
            )
            return results
        except Exception as e:
            print(f"❌ Training error: {e}")
            return None

    def export(self, format='onnx', imgsz=640):
        """Export model to different formats"""
        if not self.model_loaded:
            print("❌ No model loaded. Cannot export.")
            return None

        try:
            path = self.model.export(format=format, imgsz=imgsz)
            print(f"✅ Model exported to: {path}")
            return path
        except Exception as e:
            print(f"❌ Export error: {e}")
            return None


# Singleton instance
detector = None

def get_detector(model_path=None, conf_threshold=0.25, iou_threshold=0.45,
                  model_path_v2=None, model_path_v3=None):
    """Get or create detector singleton"""
    global detector
    if detector is None:
        detector = CropDiseaseDetector(
            model_path, conf_threshold, iou_threshold, model_path_v2, model_path_v3
        )
    return detector
