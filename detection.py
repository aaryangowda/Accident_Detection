import tensorflow as tf
import numpy as np
import cv2
import gc
import os

class AccidentDetectionModel:
    def __init__(self, model_json_path, model_weights_path):
        try:
            # Set memory growth and optimization
            tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('CPU')[0], True)
            
            # Load model architecture only first
            with open(model_json_path, 'r') as json_file:
                loaded_model_json = json_file.read()
            
            # Clear memory
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Create temporary directory for model chunks
            os.makedirs('temp_model', exist_ok=True)
            
            # Load and convert model in chunks
            self.model = tf.keras.models.model_from_json(loaded_model_json)
            
            # Load weights in chunks
            CHUNK_SIZE = 50 * 1024 * 1024  # 50MB chunks
            with open(model_weights_path, 'rb') as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    # Process chunk and clear memory
                    temp_path = 'temp_model/temp_weights.h5'
                    with open(temp_path, 'wb') as temp_file:
                        temp_file.write(chunk)
                    self.model.load_weights(temp_path, by_name=True)
                    os.remove(temp_path)
                    tf.keras.backend.clear_session()
                    gc.collect()
            
            # Convert to TFLite with dynamic range quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [
                tf.lite.Optimize.DEFAULT,
                tf.lite.Optimize.EXPERIMENTAL_SPARSITY
            ]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            
            # Representative dataset generator
            def representative_dataset():
                for _ in range(20):  # Reduced from 100 to save memory
                    data = np.random.rand(1, 250, 250, 3) * 255
                    yield [data.astype(np.float32)]
            
            converter.representative_dataset = representative_dataset
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            # Convert in chunks
            tflite_model = None
            for i in range(3):  # Try conversion up to 3 times
                try:
                    tflite_model = converter.convert()
                    break
                except Exception as e:
                    print(f"Conversion attempt {i+1} failed: {e}")
                    tf.keras.backend.clear_session()
                    gc.collect()
            
            if tflite_model is None:
                raise Exception("Failed to convert model after 3 attempts")
            
            # Create interpreter with minimal resources
            self.interpreter = tf.lite.Interpreter(
                model_content=tflite_model,
                num_threads=1,
                experimental_delegates=[]
            )
            
            # Allocate tensors
            self.interpreter.allocate_tensors()
            
            # Get details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Clean up
            del self.model
            del tflite_model
            del converter
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Remove temporary directory
            import shutil
            shutil.rmtree('temp_model', ignore_errors=True)
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise
    
    def predict_accident(self, frame):
        try:
            # Ensure minimal memory usage during prediction
            input_data = frame.astype(np.int8)
            
            # Set tensor and run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            pred = output_data[0]
            
            # Clean prediction memory
            del output_data
            gc.collect()
            
            return "Accident" if pred[0] > 0 else "No Accident", pred
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return "Error", np.array([[0.0]])
    
    def __del__(self):
        try:
            del self.interpreter
            tf.keras.backend.clear_session()
            gc.collect()
        except:
            pass
