import random
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from matplotlib import animation

class Heater:
    def __init__(self, id, pos):
        self.id = id
        self.pos = pos
        self.status = "off"
        self.power = 0.0

class OfficeZone:
    def __init__(self, id, pos, size):
        self.id = id
        self.pos = pos
        self.size = size
        self.temperature = 20.0
        self.people_count = 0
        self.target_temp = 22.0
        self.heater = None

    def add_people(self, count):
        self.people_count += count

    def remove_people(self, count):
        self.people_count = max(0, self.people_count - count)

    def update_temperature(self):
        external_temp = self.get_external_temperature()
        people_effect = self.people_count * 0.08
        
        if self.heater and self.heater.status == "on":
            heating_effect = 0.4 * (self.heater.power / 100)
        else:
            heating_effect = 0
        
        cooling_effect = (external_temp - self.temperature) * 0.03
        
        temp_change = heating_effect + people_effect + cooling_effect
        self.temperature += temp_change
        self.temperature = max(16, min(26, self.temperature))

    def get_external_temperature(self):
        hour = (simulator.step % 24)
        if 0 <= hour < 6: return 10.0
        elif 6 <= hour < 9: return 12.0
        elif 9 <= hour < 18: return 14.0
        elif 18 <= hour < 22: return 12.0
        else: return 10.0

def create_climate_system():
    temp_current = ctrl.Antecedent(np.arange(16, 27, 1), 'temp_current')
    time_of_day = ctrl.Antecedent(np.arange(0, 24, 1), 'time_of_day')
    people_presence = ctrl.Antecedent(np.arange(0, 11, 1), 'people_presence')
    heating_power = ctrl.Consequent(np.arange(0, 101, 1), 'heating_power')

    temp_current['cold'] = fuzz.trimf(temp_current.universe, [16, 16, 20])
    temp_current['comfortable'] = fuzz.trimf(temp_current.universe, [19, 22, 24])
    temp_current['hot'] = fuzz.trimf(temp_current.universe, [23, 26, 26])

    time_of_day['night'] = fuzz.trimf(time_of_day.universe, [0, 0, 7])
    time_of_day['work'] = fuzz.trimf(time_of_day.universe, [6, 9, 19])
    time_of_day['evening'] = fuzz.trimf(time_of_day.universe, [18, 22, 23])

    people_presence['none'] = fuzz.trimf(people_presence.universe, [0, 0, 1])
    people_presence['few'] = fuzz.trimf(people_presence.universe, [0, 2, 4])
    people_presence['many'] = fuzz.trimf(people_presence.universe, [3, 6, 10])

    heating_power['off'] = fuzz.trimf(heating_power.universe, [0, 0, 20])
    heating_power['low'] = fuzz.trimf(heating_power.universe, [10, 30, 50])
    heating_power['medium'] = fuzz.trimf(heating_power.universe, [40, 60, 80])
    heating_power['high'] = fuzz.trimf(heating_power.universe, [70, 100, 100])

    rules = [
        ctrl.Rule(temp_current['cold'] & people_presence['many'], heating_power['medium']),
        ctrl.Rule(temp_current['cold'] & people_presence['few'], heating_power['high']),
        ctrl.Rule(temp_current['cold'] & people_presence['none'] & time_of_day['night'], heating_power['off']),
        ctrl.Rule(temp_current['comfortable'] & people_presence['many'], heating_power['off']),
        ctrl.Rule(temp_current['comfortable'] & people_presence['few'], heating_power['low']),
        ctrl.Rule(temp_current['comfortable'] & people_presence['none'], heating_power['off']),
        ctrl.Rule(temp_current['hot'], heating_power['off']),
        ctrl.Rule(time_of_day['night'] & people_presence['none'], heating_power['off']),
        ctrl.Rule(time_of_day['work'] & temp_current['cold'], heating_power['high']),
        ctrl.Rule(people_presence['many'] & temp_current['cold'], heating_power['medium'])
    ]

    return ctrl.ControlSystem(rules)

def calculate_heating_power(zone, control_system):
    sim = ctrl.ControlSystemSimulation(control_system)
    
    hour = (simulator.step % 24)
    sim.input['temp_current'] = float(zone.temperature)
    sim.input['time_of_day'] = float(hour)
    sim.input['people_presence'] = float(zone.people_count)
    
    try:
        sim.compute()
        return sim.output['heating_power'] if 'heating_power' in sim.output else 0.0
    except:
        return 0.0

class OfficeSimulator:
    def __init__(self):
        self.step = 0
        self.zones = []
        self.heaters = []
        self.fuzzy_system = create_climate_system()
        self.setup_office()
        
    def setup_office(self):
        zone_layout = [
            {"id": "Z1", "pos": (2, 7), "size": (5, 4)},
            {"id": "Z2", "pos": (8, 7), "size": (5, 4)},
            {"id": "Z3", "pos": (2, 2), "size": (5, 4)},
            {"id": "Z4", "pos": (8, 2), "size": (5, 4)}
        ]
        
        for i, zone_data in enumerate(zone_layout):
            zone = OfficeZone(zone_data["id"], zone_data["pos"], zone_data["size"])
            heater_pos = (zone_data["pos"][0] + zone_data["size"][0]/2, 
                         zone_data["pos"][1] + zone_data["size"][1]/2)
            heater = Heater(i+1, heater_pos)
            zone.heater = heater
            self.zones.append(zone)
            self.heaters.append(heater)
    
    def simulate_people_movement(self):
        for zone in self.zones:
            if random.random() < 0.2:
                if zone.people_count > 0 and random.random() < 0.3:
                    zone.remove_people(random.randint(1, min(2, zone.people_count)))
                elif random.random() < 0.4:
                    zone.add_people(random.randint(1, 3))
    
    def update_heaters(self):
        for zone in self.zones:
            power = calculate_heating_power(zone, self.fuzzy_system)
            
            if power > 20:
                zone.heater.status = "on"
                zone.heater.power = power
            else:
                zone.heater.status = "off"
                zone.heater.power = 0
    
    def step_simulation(self):
        self.step += 1
        self.simulate_people_movement()
        
        for zone in self.zones:
            zone.update_temperature()
        
        self.update_heaters()
        
        if self.step % 10 == 0:
            self.print_status()
    
    def print_status(self):
        hour = self.step % 24
        print(f"Hour: {hour:02d}:00")
        for zone in self.zones:
            print(f"  {zone.id}: {zone.temperature:.1f}C, {zone.people_count} people, heater: {zone.heater.status}")

simulator = None

def run_office_simulation():
    global simulator
    simulator = OfficeSimulator()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 12)
    ax.set_title("Office Heating Control System")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    zone_rects = []
    heater_scatter = ax.scatter([], [], s=300, marker='s', c='red', alpha=0.6)
    temp_texts = []
    people_texts = []
    heater_texts = []
    
    for zone in simulator.zones:
        rect = plt.Rectangle(zone.pos, zone.size[0], zone.size[1], 
                           fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
        zone_rects.append(rect)
        
        temp_text = ax.text(zone.pos[0] + 1, zone.pos[1] + zone.size[1]/2 + 0.5, 
                          f"Temp: {zone.temperature:.1f}C", fontsize=9)
        temp_texts.append(temp_text)
        
        people_text = ax.text(zone.pos[0] + 1, zone.pos[1] + zone.size[1]/2,
                            f"People: {zone.people_count}", fontsize=9)
        people_texts.append(people_text)

        heater_text = ax.text(zone.heater.pos[0] - 0.5, zone.heater.pos[1] - 0.5,
                            f"H{zone.heater.id}", fontsize=10, weight='bold')
        heater_texts.append(heater_text)
    
    time_text = ax.text(0.5, 11.2, "Time: 00:00", fontsize=12)
    
    def update(frame):
        simulator.step_simulation()
        
        heater_positions = [heater.pos for heater in simulator.heaters]
        heater_colors = ['darkred' if heater.status == 'on' else 'lightcoral' 
                        for heater in simulator.heaters]
        
        heater_scatter.set_offsets(heater_positions)
        heater_scatter.set_color(heater_colors)
        
        for i, zone in enumerate(simulator.zones):
            temp_texts[i].set_text(f"Temp: {zone.temperature:.1f}C")
            people_texts[i].set_text(f"People: {zone.people_count}")
            
            color_intensity = min(1.0, max(0.0, (zone.temperature - 16) / 10))
            zone_rects[i].set_facecolor((1.0, 1.0 - color_intensity, 1.0 - color_intensity))
        
        hour = simulator.step % 24
        time_text.set_text(f"Time: {hour:02d}:00")
        
        return heater_scatter, *temp_texts, *people_texts, time_text
    
    ani = animation.FuncAnimation(fig, update, frames=240, interval=600, blit=False, repeat=True)
    ani.save("sim.gif", writer="pillow", fps=10)

if __name__ == "__main__":
    run_office_simulation()