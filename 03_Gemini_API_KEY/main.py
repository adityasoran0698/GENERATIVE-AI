from google import genai

client=genai.Client(
    api_key="AIzaSyCKvEClKq3KxmKn_KDx-mxDPzxH-6mS3G8"
)

while(True):
    user_input=input("")
    response=client.models.generate_content(
    model="gemini-2.5-flash",contents=user_input
    )
    print(response.text)




