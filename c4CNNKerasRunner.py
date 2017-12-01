from connect4.c4CNNKeras import c4CNNKeras
from scenario_runner import ScenarioRunner

runner = ScenarioRunner(c4CNNKeras())
runner.run()
