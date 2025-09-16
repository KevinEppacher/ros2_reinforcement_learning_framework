from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, EnvironmentVariable, PathJoinSubstitution

def generate_launch_description():
    logdir_arg = DeclareLaunchArgument(
        'logdir',
        default_value=PathJoinSubstitution([EnvironmentVariable('ART_DIR', default_value='/app/src/car_racing/data'), 'logs']),
        description='TensorBoard log directory'
    )
    port_arg = DeclareLaunchArgument('port', default_value='6006', description='TensorBoard port')
    open_browser_arg = DeclareLaunchArgument('open_browser', default_value='true', description='Open $BROWSER')

    tb = ExecuteProcess(
        cmd=['tensorboard', '--logdir', LaunchConfiguration('logdir'), '--bind_all', '--port', LaunchConfiguration('port')],
        output='screen'
    )

    # Open browser after TB starts
    open_browser_cmd = ExecuteProcess(
        cmd=['bash', '-lc', 'echo Opening TensorBoard...; "$BROWSER" "http://localhost:$TB_PORT"'],
        additional_env={'TB_PORT': LaunchConfiguration('port')},
        output='screen'
    )
    open_browser_when_tb_starts = RegisterEventHandler(
        OnProcessStart(
            target_action=tb,
            on_start=[TimerAction(period=1.0, actions=[open_browser_cmd], condition=IfCondition(LaunchConfiguration('open_browser')))],
        )
    )

    ld = LaunchDescription()
    ld.add_action(logdir_arg)
    ld.add_action(port_arg)
    ld.add_action(open_browser_arg)
    ld.add_action(tb)
    ld.add_action(open_browser_when_tb_starts)
    return ld