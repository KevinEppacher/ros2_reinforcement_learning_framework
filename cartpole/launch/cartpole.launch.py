from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression, EnvironmentVariable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
import os

def generate_launch_description():

    # DRL mode
    mode = DeclareLaunchArgument(
        'mode', default_value='train',
        description='Mode: train | eval'
    )

    # Namespace for rl application
    namespace = DeclareLaunchArgument(
        'namespace', default_value='cartpole_a2c',      # or use cartpole for cartpole config
        description='Namespace for the nodes'
    )

    # Open tensorboard argument
    open_tensorboard = DeclareLaunchArgument(
        'open_tensorboard', default_value='false',
        description='Whether to open TensorBoard'
    )

    # Parameters
    cartpole_params_path = os.path.join(
        get_package_share_directory('cartpole'),
        'config',
        'cartpole_rl_config.yaml'
    )

    tensorboard_launch = os.path.join(
        get_package_share_directory('rl_core'),
        'launch',
        'tensorboard.launch.py'
    )

    rl_node = Node(
        package='rl_trainers',
        executable='rl_trainers',
        name='rl_trainers',
        namespace=LaunchConfiguration('namespace'),
        parameters=[cartpole_params_path],
        output='screen',
        emulate_tty=True,
    )

    tensorboard_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(tensorboard_launch),
        launch_arguments={
            'log_dir': '/app/src/cartpole/data/logs',
            'port': '6005',
        }.items(),
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('open_tensorboard'), "' == 'true'"])),
    )

    ld = LaunchDescription()
    ld.add_action(open_tensorboard)
    ld.add_action(mode)
    ld.add_action(namespace)
    ld.add_action(rl_node)
    ld.add_action(tensorboard_node)
    return ld