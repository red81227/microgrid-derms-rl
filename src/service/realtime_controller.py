from simple_pid import PID

class RealTimeControl:

    def __init__(self, battery_limmit: float, reference_frequency: float, reference_voltage: float) -> None:
        self.battery_limmit = battery_limmit
        self.reference_frequency = reference_frequency
        self.reference_voltage = reference_voltage

    def active_power_controller(self, plan_active_power_output: float, realtime_frequence: float, Kp=0.5, Ki=0.1, Kd=0, sample_time=0.1):
        active_power_pid = PID(Kp, Ki, Kd, setpoint=self.reference_frequency)
        active_power_pid.output_limits = (-self.battery_limmit, self.battery_limmit)
        active_power_pid.sample_time = sample_time# Update every sample_time seconds
        active_power_control = active_power_pid(realtime_frequence, dt=active_power_pid.sample_time)
        final_active_power_output = plan_active_power_output + active_power_control
        return  round(final_active_power_output, 3)

    def reactive_power_controller(self, realtime_voltage: float, Kp=0.5, Ki=0.1, Kd=0, sample_time=0.1):
        reactive_power_pid = PID(Kp, Ki, Kd, setpoint=self.reference_voltage)
        reactive_power_pid.output_limits = (-self.battery_limmit, self.battery_limmit)
        reactive_power_pid.sample_time = sample_time# Update every sample_time seconds
        reactive_power_control = reactive_power_pid(realtime_voltage, dt=reactive_power_pid.sample_time)
        return  round(reactive_power_control, 3)


if __name__ == "__main__":
    battery_limmit = 500
    plan_active_power_output = 50.2
    realtime_frequence = 61.0
    reference_frequency = 60.0
    reference_voltage = 150
    realtime_voltage = 140

    real_time_control = RealTimeControl(battery_limmit, reference_frequency, reference_voltage)
    active_power_output = real_time_control.active_power_controller(plan_active_power_output, realtime_frequence)
    print(active_power_output)
    reactive_power_output = real_time_control.reactive_power_controller(realtime_voltage)
    print(reactive_power_output)
