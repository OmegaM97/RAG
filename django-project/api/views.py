from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(["POST"])
def query(request):

    question = request.data.get("question")

    return Response({
        "answer": f"You asked: {question}"
    })