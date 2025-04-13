import tensorflow as tf
import numpy as np
import cv2

class AccidentDetectionModel:
    def __init__(self, model_json_path, model_weights_path):
        # Enable mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Load and quantize model
        with open(model_json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        
        # Load the model with minimal memory usage
        tf.keras.backend.clear_session()
        self.model = tf.keras.models.model_from_json(loaded_model_json)
        self.model.load_weights(model_weights_path)
        
        # Convert model to TFLite format with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        self.interpreter = tf.lite.Interpreter(model_content=converter.convert())
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Delete the original model to free memory
        del self.model
        tf.keras.backend.clear_session()
    
    def predict_accident(self, frame):
        try:
            # Ensure input is float32 and in correct shape
            input_data = frame.astype(np.float32)
            
            # Set the input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get the output tensor
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process prediction
            pred = output_data[0]
            prediction_class = "Accident" if pred[0] > 0.5 else "No Accident"
            
            return prediction_class, pred
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return "Error", np.array([[0.0]])
            
    def __del__(self):
        # Clean up
        tf.keras.backend.clear_session()
        del self.interpreter
