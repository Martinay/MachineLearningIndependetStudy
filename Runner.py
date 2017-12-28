from TaskQueue import TaskQueue
from breakthrough.btScenario1 import btScenario1
from breakthrough.btScenario10 import btScenario10
from breakthrough.btScenario10a import btScenario10a
from breakthrough.btScenario10aa import btScenario10aa
from breakthrough.btScenario10aaaa import btScenario10aaaa
from breakthrough.btScenario10b import btScenario10b
from breakthrough.btScenario10bb import btScenario10bb
from breakthrough.btScenario10c import btScenario10c
from breakthrough.btScenario10cc import btScenario10cc
from breakthrough.btScenario10ccc import btScenario10ccc
from breakthrough.btScenario1a import btScenario1a
from connect4.c4Scenario1 import c4Scenario1
from connect4.c4Scenario10 import c4Scenario10
from connect4.c4Scenario10a import c4Scenario10a
from connect4.c4Scenario10aa import c4Scenario10aa
from connect4.c4Scenario10aaa import c4Scenario10aaa
from connect4.c4Scenario10aaaa import c4Scenario10aaaa
from connect4.c4Scenario10b import c4Scenario10b
from connect4.c4Scenario10bb import c4Scenario10bb
from connect4.c4Scenario10c import c4Scenario10c
from connect4.c4Scenario10cc import c4Scenario10cc
from connect4.c4Scenario10ccc import c4Scenario10ccc
from connect4.c4Scenario1a import c4Scenario1a
from connect4.c4Scenario2 import c4Scenario2
from connect4.c4Scenario3 import c4Scenario3
from connect4.c4Scenario4 import c4Scenario4
from connect4.c4Scenario5 import c4Scenario5
from connect4.c4Scenario6 import c4Scenario6
from connect4.c4Scenario7 import c4Scenario7
from connect4.c4Scenario8 import c4Scenario8
from connect4.c4Scenario9 import c4Scenario9
from scenario_runner import ScenarioRunner

max_epoches=30
nr_of_executions = 20
use_Gpu = True
#num_workers_threads = 1

if use_Gpu:
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    #session = tf.Session(config=config)
    set_session(tf.Session(config=config))

scenarios = [#c4Scenario1(),
             #c4Scenario1a(),
             #c4Scenario2(),
             #c4Scenario3(),
             #c4Scenario4(),
             #c4Scenario5(),
             #c4Scenario6(),
             #c4Scenario7(),
             #c4Scenario8(),
             #c4Scenario9(),
             #c4Scenario10(),
             #c4Scenario10a(),
             #c4Scenario10aa(),
             #c4Scenario10aaa(),
             #c4Scenario10aaaa(),
             #c4Scenario10b(),
             c4Scenario10bb(),
             #c4Scenario10c(),
             c4Scenario10cc(),
             c4Scenario10ccc(),
#########################breakthrough#########################
             btScenario1(),
             btScenario1a(),
             #btScenario10(),
             #btScenario10a(),
             #btScenario10aa(),
             #btScenario10aaaa(),
             #btScenario10b(),
             #btScenario10bb(),
             #btScenario10c(),
             #btScenario10cc(),
             #btScenario10ccc(),
             ]

for scenario in scenarios:
    print('start execution {}'.format(scenario.__class__.__name__))
    runner = ScenarioRunner()
    runner.init(scenario)
    runner.run(max_epoches=max_epoches, nr_of_executions=nr_of_executions)


#def runScenario(scenario, max_epoches, nr_of_executions):
#    print('Scenario {} started execution'.format(scenario.__class__.__name__))
#    runner = ScenarioRunner()
#    runner.init(scenario)
#    runner.run(max_epoches=max_epoches, nr_of_executions=nr_of_executions)


#q = TaskQueue(num_workers=num_workers_threads)

#for scenario in scenarios:
#    q.add_task(runScenario, scenario, max_epoches=max_epoches, nr_of_executions=nr_of_executions)
#    print('Scenario {} added'.format(str(scenario.__class__.__name__)))

#q.join()
print('finished')