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

    # Namespace for rl application
    namespace = DeclareLaunchArgument(
        'namespace', default_value='car_racing',
        description='Namespace for the nodes'
    )

    # Open tensorboard argument
    open_tensorboard = DeclareLaunchArgument(
        'open_tensorboard', default_value='false',
        description='Whether to open TensorBoard'
    )

    # Parameters
    car_racing_params_path = os.path.join(
        get_package_share_directory('car_racing'),
        'config',
        'car_racing_rl_config.yaml'
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
        parameters=[car_racing_params_path],
        output='screen',
        emulate_tty=True,
    )

    tensorboard_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(tensorboard_launch),
        launch_arguments={
            'log_dir': '/app/src/car_racing/data/logs',
            'port': '6005',
        }.items(),
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('open_tensorboard'), "' == 'true'"])),
    )

    ld = LaunchDescription()
    ld.add_action(open_tensorboard)
    ld.add_action(namespace)
    ld.add_action(rl_node)
    ld.add_action(tensorboard_node)
    return ld