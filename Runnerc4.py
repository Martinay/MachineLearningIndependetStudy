from connect4.c4Scenario1 import c4Scenario1
from connect4.c4Scenario10 import c4Scenario10
from connect4.c4Scenario10a import c4Scenario10a
from connect4.c4Scenario10aa import c4Scenario10aa
from connect4.c4Scenario10aaa import c4Scenario10aaa
from connect4.c4Scenario10aaaa import c4Scenario10aaaa
from connect4.c4Scenario2 import c4Scenario2
from connect4.c4Scenario3 import c4Scenario3
from connect4.c4Scenario4 import c4Scenario4
from connect4.c4Scenario5 import c4Scenario5
from connect4.c4Scenario6 import c4Scenario6
from connect4.c4Scenario7 import c4Scenario7
from connect4.c4Scenario8 import c4Scenario8
from connect4.c4Scenario9 import c4Scenario9
from scenario_runner import ScenarioRunner

input = 14
if input == 1:
    runner = ScenarioRunner(c4Scenario1())
elif input == 2:
    runner = ScenarioRunner(c4Scenario2())
elif input == 3:
    runner = ScenarioRunner(c4Scenario3())
elif input == 4:
    runner = ScenarioRunner(c4Scenario4())
elif input == 5:
    runner = ScenarioRunner(c4Scenario5())
elif input == 6:
    runner = ScenarioRunner(c4Scenario6())
elif input == 7:
    runner = ScenarioRunner(c4Scenario7())
elif input == 8:
    runner = ScenarioRunner(c4Scenario8())
elif input == 9:
    runner = ScenarioRunner(c4Scenario9())
elif input == 10:
    runner = ScenarioRunner(c4Scenario10())
elif input == 11:
    runner = ScenarioRunner(c4Scenario10a())
elif input == 12:
    runner = ScenarioRunner(c4Scenario10aa())
elif input == 13:
    runner = ScenarioRunner(c4Scenario10aaa())
elif input == 14:
    runner = ScenarioRunner(c4Scenario10aaaa())

runner.run()
