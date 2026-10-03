#!/bin/bash

# TWIST2+IsaacLab enhanced data recording system - launcher
#
# Automatically start all required components
# Usage:bash start_enhanced_recording.sh

set -e  # Exit immediately on error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print a colored message
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
}

# Check dependencies
check_dependencies() {
    print_header "检查系统依赖"

    # Check Redis
    if ! command -v redis-cli &> /dev/null; then
        print_error "Redis未安装，请先安装Redis"
        exit 1
    fi

    if ! redis-cli ping &> /dev/null; then
        print_error "Redis未运行，正在启动..."
        sudo systemctl start redis || redis-server --daemonize yes
        sleep 2
        if redis-cli ping &> /dev/null; then
            print_success "Redis启动成功"
        else
            print_error "Redis启动失败"
            exit 1
        fi
    else
        print_success "Redis运行正常"
    fi

    # Check the Python environment
    if ! command -v python3 &> /dev/null; then
        print_error "Python3未安装"
        exit 1
    fi
    print_success "Python3已安装: $(python3 --version)"
}

# Configure parameters
configure_params() {
    print_header "配置参数"

    # Default parameters
    DEFAULT_ROBOT_IP="192.168.123.164"
    DEFAULT_TASK_NAME="demo_$(date +%Y%m%d_%H%M%S)"
    DEFAULT_FREQUENCY="30"
    DEFAULT_DATA_FOLDER="./twist2_demonstration_smplx"

    # Interactive configuration or use defaults
    read -p "机器人IP地址 [${DEFAULT_ROBOT_IP}]: " ROBOT_IP
    ROBOT_IP=${ROBOT_IP:-$DEFAULT_ROBOT_IP}

    read -p "任务名称 [${DEFAULT_TASK_NAME}]: " TASK_NAME
    TASK_NAME=${TASK_NAME:-$DEFAULT_TASK_NAME}

    read -p "录制频率(Hz) [${DEFAULT_FREQUENCY}]: " FREQUENCY
    FREQUENCY=${FREQUENCY:-$DEFAULT_FREQUENCY}

    read -p "数据保存路径 [${DEFAULT_DATA_FOLDER}]: " DATA_FOLDER
    DATA_FOLDER=${DATA_FOLDER:-$DEFAULT_DATA_FOLDER}

    echo ""
    print_info "配置摘要:"
    echo "  机器人IP: ${ROBOT_IP}"
    echo "  任务名称: ${TASK_NAME}"
    echo "  录制频率: ${FREQUENCY} Hz"
    echo "  数据路径: ${DATA_FOLDER}"
    echo ""

    read -p "是否继续? (y/n) [y]: " CONFIRM
    CONFIRM=${CONFIRM:-y}
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        print_info "已取消"
        exit 0
    fi
}

# Check required scripts
check_scripts() {
    print_header "检查脚本文件"

    TWIST2_DIR="/home/hcl4070-1/Desktop/taowen/projects/TWIST2"
    ISAACLAB_DIR="/home/hcl4070-1/Desktop/taowen/projects/isaaclab_twist2_g1"

    # Check the TWIST2 teleoperation script
    TELEOP_SCRIPT="${TWIST2_DIR}/deploy_real/xrobot_teleop_to_robot_w_hand.py"
    if [[ ! -f "$TELEOP_SCRIPT" ]]; then
        print_error "遥操作脚本不存在: $TELEOP_SCRIPT"
        exit 1
    fi
    print_success "遥操作脚本: $TELEOP_SCRIPT"

    # Check the data recording script
    RECORD_SCRIPT="${TWIST2_DIR}/deploy_real/server_data_record_with_third_smplx_qpos.py"
    if [[ ! -f "$RECORD_SCRIPT" ]]; then
        print_error "数据记录脚本不存在: $RECORD_SCRIPT"
        exit 1
    fi
    print_success "数据记录脚本: $RECORD_SCRIPT"

    # Check the IsaacLab sim_main entry point
    ISAACLAB_SCRIPT="${ISAACLAB_DIR}/sim_main.py"
    if [[ ! -f "$ISAACLAB_SCRIPT" ]]; then
        print_warning "IsaacLab脚本不存在: $ISAACLAB_SCRIPT"
        print_warning "你需要手动启动IsaacLab仿真"
    else
        print_success "IsaacLab脚本: $ISAACLAB_SCRIPT"
    fi
}

# Check whether IsaacLab is running
check_isaaclab() {
    print_header "检查IsaacLab状态"

    # Check the image-server port
    if nc -zv localhost 5555 2>&1 | grep -q succeeded; then
        print_success "Front camera 服务器 (Port 5555) 运行正常"
    else
        print_warning "Front camera 服务器 (Port 5555) 未检测到"
        print_info "请确保IsaacLab仿真正在运行并启动了图像服务器"
    fi

    if nc -zv localhost 5556 2>&1 | grep -q succeeded; then
        print_success "World camera 服务器 (Port 5556) 运行正常"
    else
        print_warning "World camera 服务器 (Port 5556) 未检测到"
        print_info "请确保IsaacLab仿真正在运行并启动了图像服务器"
    fi

    echo ""
    print_info "如果IsaacLab未运行，请在另一个终端中启动:"
    echo "  cd ${ISAACLAB_DIR}"
    echo "  ./run.sh"
    echo ""
    read -p "IsaacLab是否已经运行? (y/n) [y]: " ISAACLAB_RUNNING
    ISAACLAB_RUNNING=${ISAACLAB_RUNNING:-y}
    if [[ ! "$ISAACLAB_RUNNING" =~ ^[Yy]$ ]]; then
        print_error "请先启动IsaacLab仿真"
        exit 1
    fi
}

# Start the teleoperation system
start_teleop() {
    print_header "启动TWIST2遥操作系统"

    cd "${TWIST2_DIR}"

    # Check whether it is already running
    if pgrep -f "xrobot_teleop_to_robot_w_hand.py" > /dev/null; then
        print_success "遥操作系统已在运行"
        return 0
    fi

    print_info "在新终端启动遥操作系统..."

    # Launch with gnome-terminal or xterm
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash -c "
            cd ${TWIST2_DIR};
            echo '启动TWIST2遥操作系统...';
            python deploy_real/xrobot_teleop_to_robot_w_hand.py --robot unitree_g1;
            read -p '按Enter键关闭...'
        " &
        TELEOP_PID=$!
        print_success "遥操作系统启动中（新终端）..."
    elif command -v xterm &> /dev/null; then
        xterm -e "
            cd ${TWIST2_DIR};
            echo '启动TWIST2遥操作系统...';
            python deploy_real/xrobot_teleop_to_robot_w_hand.py --robot unitree_g1;
            read -p '按Enter键关闭...'
        " &
        TELEOP_PID=$!
        print_success "遥操作系统启动中（新终端）..."
    else
        print_warning "无法自动打开新终端"
        print_info "请手动在新终端运行:"
        echo "  cd ${TWIST2_DIR}"
        echo "  python deploy_real/xrobot_teleop_to_robot_w_hand.py --robot unitree_g1"
        echo ""
        read -p "启动完成后按Enter继续..."
    fi

    # Wait for the teleoperation system to start
    print_info "等待遥操作系统启动..."
    sleep 5

    # Check whether SMPL-X data is available
    for i in {1..10}; do
        if redis-cli exists smplx_data_unitree_g1_with_hands | grep -q 1; then
            print_success "SMPLX数据已就绪"
            break
        fi
        if [ $i -eq 10 ]; then
            print_warning "SMPLX数据未检测到，但继续执行..."
        fi
        sleep 1
    done
}

# Start data recording
start_recording() {
    print_header "启动数据记录系统"

    cd "${TWIST2_DIR}"

    print_info "启动参数:"
    echo "  --robot_ip ${ROBOT_IP}"
    echo "  --task_name ${TASK_NAME}"
    echo "  --frequency ${FREQUENCY}"
    echo "  --data_folder ${DATA_FOLDER}"
    echo ""

    print_info "控制说明:"
    echo "  左手柄 key_two    : 开始/停止录制"
    echo "  左手柄 axis_click : 退出程序"
    echo ""

    print_success "正在启动数据记录..."

    python deploy_real/server_data_record_with_third_smplx_qpos.py \
        --robot_ip "${ROBOT_IP}" \
        --task_name "${TASK_NAME}" \
        --frequency "${FREQUENCY}" \
        --data_folder "${DATA_FOLDER}"
}

# Cleanup function
cleanup() {
    print_header "清理资源"

    # Add cleanup logic here
    # For example, stop the processes started by this script

    print_success "清理完成"
}

# Run main function
main() {
    # Handle Ctrl+C
    trap cleanup EXIT

    print_header "TWIST2+IsaacLab 增强数据记录系统"

    # 1. Check dependencies
    check_dependencies

    # 2. Configure parameters
    configure_params

    # 3. Check scripts
    check_scripts

    # 4. Check IsaacLab
    check_isaaclab

    # 5. Start teleoperation (optional)
    read -p "是否需要启动TWIST2遥操作系统? (y/n) [y]: " START_TELEOP
    START_TELEOP=${START_TELEOP:-y}
    if [[ "$START_TELEOP" =~ ^[Yy]$ ]]; then
        start_teleop
    else
        print_info "跳过遥操作系统启动（假设已运行）"
    fi

    # 6. Start data recording
    start_recording

    print_success "所有组件已关闭"
}

# Run main function
main "$@"
