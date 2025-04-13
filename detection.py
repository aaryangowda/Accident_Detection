import tensorflow as tf
import numpy as np
import cv2
import gc

class AccidentDetectionModel:
    def __init__(self, model_json_path, model_weights_path):
        try:
            # Load and quantize model
            with open(model_json_path, 'r') as json_file:
                loaded_model_json = json_file.read()
            
            # Clear any existing sessions
            tf.keras.backend.clear_session()
            
            # Load model with minimal memory
            self.model = tf.keras.models.model_from_json(loaded_model_json)
            self.model.load_weights(model_weights_path)
            
            # Convert to TFLite with full integer quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            # Representative dataset for quantization
            def representative_dataset():
                for _ in range(100):
                    data = np.random.rand(1, 250, 250, 3) * 255
                    yield [data.astype(np.float32)]
            
            converter.representative_dataset = representative_dataset
            tflite_model = converter.convert()
            
            # Create interpreter with reduced tensor arena size
            self.interpreter = tf.lite.Interpreter(
                model_content=tflite_model,
                num_threads=1
            )
            
            # Allocate only necessary tensors
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get quantization parameters
            self.input_scale, self.input_zero_point = self.input_details[0]["quantization"]
            self.output_scale, self.output_zero_point = self.output_details[0]["quantization"]
            
            # Delete the original model and clear session
            del self.model
            del tflite_model
            tf.keras.backend.clear_session()
            gc.collect()
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise
    
    def predict_accident(self, frame):
        try:
            # Quantize input
            input_data = frame / self.input_scale + self.input_zero_point
            input_data = input_data.astype(np.int8)
            
            # Set the input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get the output tensor and dequantize
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            output_data = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale
            
            # Process prediction
            pred = output_data[0]
            prediction_class = "Accident" if pred[0] > 0.5 else "No Accident"
            
            return prediction_class, pred
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return "Error", np.array([[0.0]])
            
    def __del__(self):
        try:
            # Clean up
            del self.interpreter
            tf.keras.backend.clear_session()
            gc.collect()
        except:
            pass
