from invoke import task
import yaml

@task
def copy(c, base, postfix):
    """
    Make a copy of a chart dir to run multiple times in parallel without naming collisions. Best practice is to manually modify the chartdir once, e.g. create maskrcnn_16x1_fix, and then create a new copy for each run.

    Create a valid chartdir, maskrcnn_16x1_fix

    I want to run this experiment five times, but reusing this same chartdir multiple times will fail due to naming conflicts. Instead use this tool to create copies of the chartdir that do not have naming conflicts.

    invoke copy maskrcnn_16x1_fix run1
    invoke copy maskrcnn_16x1_fix run2
    invoke copy maskrcnn_16x1_fix run3
    invoke copy maskrcnn_16x1_fix run4
    invoke copy maskrcnn_16x1_fix run5

    Now you have

    maskrcnn_16x1_fix/
    maskrcnn_16x1_fix_run1/
    maskrcnn_16x1_fix_run2/
    maskrcnn_16x1_fix_run3/
    maskrcnn_16x1_fix_run4/
    maskrcnn_16x1_fix_run5/

    All of the copies will have the order of their yaml fields shuffled, so it's useful to maskrcnn_16x1_fix/ as your reference chartdir.

    Now trigger five runs with

    helm install --name maskrcnn-16x1-fix-run1 ./maskrcnn_16x1_fix_run1/
    helm install --name maskrcnn-16x1-fix-run2 ./maskrcnn_16x1_fix_run2/
    helm install --name maskrcnn-16x1-fix-run3 ./maskrcnn_16x1_fix_run3/
    helm install --name maskrcnn-16x1-fix-run4 ./maskrcnn_16x1_fix_run4/
    helm install --name maskrcnn-16x1-fix-run5 ./maskrcnn_16x1_fix_run5/

    """
    print(f'Creating chartdef {base}_{postfix}')
    while base.endswith("/"):
        base = base[:-1]
    c.run(f'cp -r ./{base} ./{base}_{postfix}')

    mpioperator_values_path = f'./{base}_{postfix}/charts/mpi-operator/values.yaml'
    maskrcnn_values_path = f'./{base}_{postfix}/values.yaml'

    with open(mpioperator_values_path, 'r') as f:
        mpioperator_vals = yaml.safe_load(f)

    with open(maskrcnn_values_path, 'r') as f:
        maskrcnn_vals = yaml.safe_load(f)

    # Overwrite
    mpioperator_vals["mpioperator"]["name"] += f'-{postfix}'
    maskrcnn_vals["maskrcnn"]["name"] += f'-{postfix}'

    with open(mpioperator_values_path, 'w') as f:
        yaml.dump(mpioperator_vals, f)

    with open(maskrcnn_values_path, 'w') as f:
        yaml.dump(maskrcnn_vals, f)



