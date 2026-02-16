import requests


# https://thingspeak.mathworks.com/channels/3258476/private_show
# Channel States:  https://thingspeak.mathworks.com/channels/3258476
# RESR API:  https://www.mathworks.com/help/thingspeak/rest-api.html
def _things_speak_url(period, ):
    """https://thingspeak.mathworks.com/channels/3258476/api_keys"""
    write_api_key = "87YUBRFXK5VZOLJG"
    # GET https://api.thingspeak.com/update?api_key=87YUBRFXK5VZOLJG&field1=0
    swing_angle = max_angle(self.max_l_pendulum, self.max_r_pendulum)
    sewing_distance = calculate_distance(self.max_l_pendulum, self.max_r_pendulum)
    period = filtered_mean(self.pendulum_period)
    width = filtered_mean(self.pendulum_width)
    url = (f"https://api.thingspeak.com/update?api_key={write_api_key}"
           f"&field1={self.max_l_pendulum[1]}&field2={self.max_r_pendulum[1]}&field3={swing_angle}"
           f"&field4={sewing_distance}&field5={period}&field6={width}")
    return url


def thingsspeak_post(period):
    try:
        response = requests.post(_things_speak_url(period))
        if response.status_code == 200:
            print(f"Data sent successfully. Response: {response.text}")
        else:
            print(f"Failed to send data. Status code: {response.status_code}, Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Connection failed: {e}")