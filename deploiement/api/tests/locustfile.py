from locust import HttpUser, task, between

# Test sur la webapp
#class SegmentUser(HttpUser):
#    wait_time = between(0.5, 0.5)  # 0.5 sec entre chaque envoi

#    @task
#    def predict_image(self):
#        with open("test.png", "rb") as image_file:
#            files = {
#                "file": ("test.png", image_file, "image/png")
#            }
#            response = self.client.post("/", files=files)
#            print(f"Status: {response.status_code}")

# Test sur l'API
from locust import HttpUser, task, between

class FastAPISegmentUser(HttpUser):
    wait_time = between(0.1, 0.1)

    @task
    def predict_direct_api(self):
        with open("test.png", "rb") as image_file:
            files = {
                "file": ("test.png", image_file, "image/png")
            }
            response = self.client.post("/predict", files=files)
            print(f"Status: {response.status_code}")

