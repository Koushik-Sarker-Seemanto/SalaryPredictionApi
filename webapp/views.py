from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import numpy as np
import pickle
import os


def predict_method(data_list):
    try:
        settings_dir = os.path.dirname(__file__)
        root = os.path.abspath(os.path.dirname(settings_dir))
        pretrained_model_path = os.path.join(root, 'webapp\\training\\finalized_model.sav')

        print('################## Pretrained Model Path ##################')
        print(pretrained_model_path)
        loaded_model = pickle.load(open(pretrained_model_path, 'rb'))

        print('################## Pretrained Model Load Complete ################## ', loaded_model)

        arr = np.array([data_list])
        predicted_salary = loaded_model.predict(arr)
        print('############################## Prediction: ', predicted_salary[0])
        return predicted_salary[0]
    except:
        print('Invalid prediction')
        raise Exception('Invalid prediction')


@api_view(['POST'])
def predict(request):
    print('started!!!!!!!!!!!!!!!!')
    try:
        request_data = request.data
        print(request_data)
        lst = [request_data['java'], request_data['c#'], request_data['python'], request_data['c'], request_data['c++'],
               request_data['html'], request_data['javascript'], request_data['rubby'], request_data['css'],
               request_data['go'], request_data['swift'], request_data['php'], request_data['kotlin'],
               request_data['dart']]
        result = predict_method(lst)
        result = round(result, 2)
        print(result)
        if result < 10000:
            result = 0
        response = {"result": result, "error": None}
        return Response(response, status=status.HTTP_200_OK)
    except:
        response = {"result": None, "error": "Internal Server Error"}
        return Response(response, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def health(request):
    print('started!!!!!!!!!!!!!!!!')
    return Response("healthy", status=status.HTTP_200_OK)
