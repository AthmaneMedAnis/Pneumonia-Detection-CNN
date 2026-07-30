import json 
import base64
import numpy as np
from PIL import Image
import io
from ai_edge_litert.interpreter import Interpreter
interpreter = Interpreter(model_path='./models/pneumonia_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        image_data = base64.b64decode(body['image_base64'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((224, 224))

        img_array = np.array(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        result = "Suspected Pneumonia" if prediction >= 0.5 else 'Normal'
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'diagnostic': result,
                'pneumonie_probability': float(prediction)
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }