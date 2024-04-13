import numpy as np
from scipy.optimize import fsolve
import json

# Physical constants
g = 9.81  # m/s^2
rho = 1.293  # kg/m^3


class DroneModel:

    def __init__(self, input_file, airspeed=20):
        # Read drone model
        params = json.load(open(input_file))
        self.airspeed = airspeed
        self.m_body = params["weight"]["body"]
        self.m_battery = params["weight"]["battery"]
        self.m_payload = params["weight"]["payload"]
        self.f_d = 0.5 * rho * airspeed ** 2 * params["drag_coefficient"]["body"] * params["front_area"]["body"] + \
                   0.5 * rho * airspeed ** 2 * params["drag_coefficient"]["battery"] * params["front_area"]["battery"] + \
                   0.5 * rho * airspeed ** 2 * params["drag_coefficient"]["payload"] * params["front_area"]["payload"]
        self.engine_components = np.pi * params["rotors"]["number"] * params["rotors"]["diameter"] ** 2 * rho
        self.power_efficiency = params["power_efficiency"]
        self.max_battery = params["battery"]["capacity"] * params["battery"]["safe_discharge"]

    def pitch_angle(self, payload):
        return np.arctan(self.f_d / (self.m_body + self.m_battery + self.m_payload + payload) / g)

    def thrust_force(self, payload):
        return (self.m_body + self.m_battery + self.m_payload + payload) * g + self.f_d

    def implicit_velocity(self, payload, thrust=None, pitch=None):
        if thrust is None:
            thrust = self.thrust_force(payload)
        if pitch is None:
            pitch = self.pitch_angle(payload)
        T2 = 2 * thrust
        v_cos = (self.airspeed * np.cos(pitch)) ** 2
        v_sin = self.airspeed * np.sin(pitch)

        def f(v_i):
            return (T2 / (self.engine_components * np.sqrt(v_cos + (v_i + v_sin) ** 2))) - v_i

        return fsolve(f, np.array([self.airspeed]))[0]

    def power_consumption(self, payload, thrust=None, pitch=None):
        if thrust is None:
            thrust = self.thrust_force(payload)
        if pitch is None:
            pitch = self.pitch_angle(payload)
        return thrust * (self.implicit_velocity(payload, thrust, pitch) + self.airspeed * np.sin(pitch)) / self.power_efficiency

    def max_flight_time(self, payload, thrust=None, pitch=None):
        if thrust is None:
            thrust = self.thrust_force(payload)
        if pitch is None:
            pitch = self.pitch_angle(payload)
        return self.max_battery / self.power_consumption(payload, thrust, pitch)


if __name__ == '__main__':
    model = DroneModel("data/drone_models/octocopter.json")
