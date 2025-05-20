#!/bin/bash
# LTX-Video训练流水线脚本 - Linux版
# 基于minimal_ui.py构建的命令行版本

# 设置UTF-8编码
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# 设置环境变量
export PYTHONIOENCODING=utf-8
# 使用本地模型时保持离线模式
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 定义本地LLaVA模型路径
LLAVA_MODEL_PATH="$SCRIPT_DIR/models/LLaVA-NeXT-Video-7B-hf"

# 设置Python路径
SCRIPT_DIR=$(dirname "$(realpath "$0")")
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/src:$PYTHONPATH"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # 无颜色

# 默认值
PROJECT_DIR=$(dirname "$(readlink -f "$0")")
SCRIPTS_DIR="$PROJECT_DIR/scripts"
DATA_DIR="$PROJECT_DIR/train_date"
CONFIGS_DIR="$PROJECT_DIR/configs"
DIFFUSERS_MODEL_PATH="$PROJECT_DIR/models/LTX-Video-0.9.7-diffusers"

# 检查LLaVA模型是否已下载
if [ ! -d "$LLAVA_MODEL_PATH" ]; then
    echo -e "${YELLOW}警告: LLaVA-NeXT-Video-7B-hf模型未下载。${NC}"
    echo -e "${YELLOW}请先运行 'python download_llava_model.py' 下载模型${NC}"
    echo -e "${YELLOW}或者在标注时选择使用Moonshot API${NC}"
fi

# 预设分辨率列表
declare -a RESOLUTIONS_DIMS=(
    "[FANG] 320x320"
    "[FANG] 384x384"
    "[FANG] 416x416"
    "[FANG] 512x512"
    "[FANG] 640x640"
    "[FANG] 768x768"
    "[FANG] 1024x1024"
    "[HENG] 480x320"
    "[HENG] 640x384"
    "[HENG] 704x416"
    "[HENG] 768x448"
    "[HENG] 960x544"
    "[HENG] 1024x576"
    "[HENG] 1280x736"
    "[SHU] 320x480"
    "[SHU] 384x640"
    "[SHU] 416x704"
    "[SHU] 448x768"
    "[SHU] 576x1024"
    "[SHU] 736x1280"
)

# 帧数列表
declare -a FRAME_COUNTS=(
    "25"
    "49"
    "73"
    "97"
    "121"
    "145"
    "169"
    "193"
    "241"
)

# 读取配置文件
function load_config_files() {
    echo -e "${BLUE}读取配置文件...${NC}"
    
    # 清空配置文件列表
    CONFIG_FILES=()
    CONFIG_NAMES=()
    
    # 读取默认配置文件
    if [ -d "$CONFIGS_DIR" ]; then
        for file in "$CONFIGS_DIR"/*.yaml; do
            if [ -f "$file" ]; then
                name=$(basename "$file" .yaml)
                CONFIG_FILES+=("$file")
                CONFIG_NAMES+=("$name")
                echo -e "找到配置: ${CYAN}$name${NC}"
            fi
        done
    fi
    
    # 读取用户自定义配置文件
    USER_CONFIGS_DIR="$PROJECT_DIR/user_configs"
    if [ -d "$USER_CONFIGS_DIR" ]; then
        for file in "$USER_CONFIGS_DIR"/*.yaml; do
            if [ -f "$file" ]; then
                name=$(basename "$file" .yaml)
                display_name="[自定义] $name"
                CONFIG_FILES+=("$file")
                CONFIG_NAMES+=("$display_name")
                echo -e "找到用户自定义配置: ${CYAN}$display_name${NC}"
            fi
        done
    fi
    
    if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
        echo -e "${RED}错误: 未找到配置文件!${NC}"
        exit 1
    fi
}

# 运行命令并显示输出
function run_command() {
    cmd=("$@")
    echo -e "${YELLOW}执行命令: ${cmd[*]}${NC}"
    echo "请等待..."
    echo
    
    # 检查命令类型以确定是否需要实时输出
    cmd_str="${cmd[*]}"
    
    # 判断命令类型
    if [[ "$cmd_str" == *"preprocess_zero_workers.py"* ]] || [[ "$cmd_str" == *"fix_resolution_wrapper.py"* ]] || 
       [[ "$cmd_str" == *"train.py"* ]] || [[ "$cmd_str" == *"caption_videos.py"* ]]; then
        # 需要实时输出的命令，直接执行
        "${cmd[@]}"
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            if [[ "$cmd_str" == *"preprocess_zero_workers.py"* ]] || [[ "$cmd_str" == *"fix_resolution_wrapper.py"* ]]; then
                echo -e "${GREEN}预处理命令执行成功！${NC}"
            elif [[ "$cmd_str" == *"train.py"* ]]; then
                echo -e "${GREEN}训练命令执行成功！${NC}"
            elif [[ "$cmd_str" == *"caption_videos.py"* ]]; then
                echo -e "${GREEN}标注命令执行成功！${NC}"
            else
                echo -e "${GREEN}命令执行成功！${NC}"
            fi
        else
            echo -e "${RED}命令执行失败，返回代码: $exit_code${NC}"
        fi
    else
        # 其他命令，捕获输出
        output=$("${cmd[@]}" 2>&1)
        exit_code=$?
        
        echo "$output"
        echo
        
        if [ $exit_code -eq 0 ]; then
            echo -e "${GREEN}命令执行成功！${NC}"
        else
            echo -e "${RED}命令执行失败，返回代码: $exit_code${NC}"
        fi
    fi
    
    return $exit_code
}

# 从带有前缀标识的分辨率字符串中提取实际分辨率
function extract_dims() {
    local dims_string="$1"
    
    # 如果分辨率字符串包含方向标识前缀，去除前缀
    if [[ "$dims_string" == *"["*"]"* ]]; then
        # 找到最后一个方括号位置并取其后的内容
        dims_string=$(echo "$dims_string" | sed 's/.*\] //')
    fi
    
    echo "$dims_string"
}

# 从数据集名称中提取触发词
function extract_trigger_word() {
    local basename="$1"
    
    # 如果是"XXX_scenes"格式，移除"_scenes"后缀
    if [[ "$basename" == *"_scenes" ]]; then
        basename=${basename%_scenes}
    fi
    
    # 如果包含下划线，取第一段
    if [[ "$basename" == *"_"* ]]; then
        echo "${basename%%_*}"
        return
    fi
    
    # 如果包含空格，取第一段
    if [[ "$basename" == *" "* ]]; then
        echo "${basename%% *}"
        return
    fi
    
    # 否则使用整个名称
    echo "$basename"
}

# 检查数据集位置
function check_dataset_location() {
    local basename="$1"
    
    # 检查train_date/basename格式
    standard_path="$DATA_DIR/$basename"
    if [ -d "$standard_path" ]; then
        echo "$standard_path"
        return
    fi
    
    # 检查basename_raw格式
    raw_path="${basename}_raw"
    raw_full_path="$PROJECT_DIR/$raw_path"
    if [ -d "$raw_full_path" ]; then
        echo "$raw_full_path"
        return
    fi
    
    # 都不存在
    echo ""
}

# 获取预处理数据的绝对路径
function get_preprocessed_path() {
    local basename="$1"
    
    # 首先尝试找到数据集位置
    dataset_path=$(check_dataset_location "$basename")
    
    # 如果我们找到了数据集路径，尝试不同的预处理数据位置
    if [ -n "$dataset_path" ]; then
        # 首先检查.precomputed目录
        precomputed_path="$dataset_path/.precomputed"
        if [ -d "$precomputed_path" ]; then
            if [ -d "$precomputed_path/latents" ] && [ -d "$precomputed_path/conditions" ]; then
                echo "找到.precomputed目录下的预处理数据: $precomputed_path"
                echo "$precomputed_path"
                return
            fi
        fi
        
        # 检查项目名_scenes目录
        scenes_path="$dataset_path/${basename}_scenes"
        if [ -d "$scenes_path" ]; then
            if [ -d "$scenes_path/latents" ] && [ -d "$scenes_path/conditions" ]; then
                echo "找到${basename}_scenes目录下的预处理数据: $scenes_path"
                echo "$scenes_path"
                return
            fi
        fi
        
        # 检查scenes目录
        alt_scenes_path="$dataset_path/scenes"
        if [ -d "$alt_scenes_path" ]; then
            if [ -d "$alt_scenes_path/latents" ] && [ -d "$alt_scenes_path/conditions" ]; then
                echo "找到scenes目录下的预处理数据: $alt_scenes_path"
                echo "$alt_scenes_path"
                return
            fi
        fi
        
        # 如果在数据集路径下找不到预处理数据，但找到了.precomputed目录，则返回该路径
        if [ -d "$precomputed_path" ]; then
            echo "找到.precomputed目录，但可能不完整: $precomputed_path"
            echo "$precomputed_path"
            return
        fi
        
        # 如果找到了数据集但没有预处理数据，则返回.precomputed路径（可能还未创建）
        echo "数据集存在但无预处理数据，将使用默认.precomputed路径"
        echo "$dataset_path/.precomputed"
        return
    fi
    
    # 如果找不到数据集路径，使用默认路径
    default_path="$DATA_DIR/$basename/.precomputed"
    echo "无法找到预处理数据，将使用默认路径: $default_path"
    echo "$default_path"
}

# 显示标题
function show_title() {
    clear
    echo -e "${BLUE}=========================================================${NC}"
    echo -e "${BLUE}                  LTX-Video-Trainer                      ${NC}"
    echo -e "${BLUE}                     Linux 脚本版                        ${NC}"
    echo -e "${BLUE}=========================================================${NC}"
    echo
}

# 主要流水线函数
function run_pipeline() {
    # 参数
    local basename="$1"
    local dims="$2"
    local frames="$3"
    local config_name="$4"
    local rank="$5"
    local split_scenes="$6"
    local caption="$7"
    local preprocess="$8"
    local only_preprocess="$9"
    local add_trigger="${10}"
    local caption_method="${11}"
    
    # 提取实际分辨率，去除标识前缀
    dimensions=$(extract_dims "$dims")
    
    # 组合分辨率和帧数
    resolution="${dimensions}x${frames}"
    
    # 如果在只预处理模式下或config_name为空，跳过配置文件验证
    if [ "$only_preprocess" = true ] || [ -z "$config_name" ]; then
        echo "只执行预处理模式，跳过配置文件验证"
        temp_config_path=""
    else
        # 正常验证配置文件
        config_template=""
        for i in "${!CONFIG_NAMES[@]}"; do
            if [ "${CONFIG_NAMES[$i]}" = "$config_name" ]; then
                config_template="${CONFIG_FILES[$i]}"
                break
            fi
        done
        
        if [ -z "$config_template" ]; then
            echo -e "${RED}错误: 找不到配置 $config_name${NC}"
            return 1
        fi
        
        echo -e "使用配置模板: ${CYAN}$config_template${NC} 创建临时配置文件"
    fi
    
    # 只有在需要训练时才检查模型文件
    model_path=""
    if [ "$only_preprocess" = false ] && [ -n "$config_name" ]; then
        if [ -d "$DIFFUSERS_MODEL_PATH" ]; then
            echo -e "使用diffusers格式模型: ${CYAN}$DIFFUSERS_MODEL_PATH${NC}"
            model_path="$DIFFUSERS_MODEL_PATH"
        else
            echo -e "${RED}错误: 未找到diffusers模型: $DIFFUSERS_MODEL_PATH${NC}"
            return 1
        fi
    elif [ "$only_preprocess" = true ]; then
        echo "只执行预处理模式，跳过模型文件检查"
    fi
    
    # 检查数据集路径
    dataset_path=$(check_dataset_location "$basename")
    if [ -z "$dataset_path" ]; then
        echo -e "${RED}错误: 未找到数据集 '$basename'${NC}"
        echo -e "${RED}请确保数据位于 'train_date/$basename' 或 '${basename}_raw' 目录${NC}"
        return 1
    fi
    
    # ======== 步骤1: 分场景步骤（可选）========
    echo -e "\n${BLUE}=== 步骤1: 分场景处理 ===${NC}"
    
    # 原始视频应该在数据集下的raw_videos目录
    raw_videos_dir="$dataset_path/raw_videos"
    # 场景输出目录应该在数据集下的scenes目录
    scenes_dir="$dataset_path/scenes"
    
    # 检查原始视频目录下是否有视频文件
    raw_videos=()
    if [ -d "$raw_videos_dir" ]; then
        while IFS= read -r -d '' file; do
            raw_videos+=("$file")
        done < <(find "$raw_videos_dir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) -print0)
    fi
    
    # 检查数据集根目录下是否有视频文件
    video_files=()
    while IFS= read -r -d '' file; do
        # 过滤掉raw_videos和scenes目录下的文件
        if [[ "$file" != *"/raw_videos/"* ]] && [[ "$file" != *"/scenes/"* ]]; then
            video_files+=("$file")
        fi
    done < <(find "$dataset_path" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) -print0)
    
    if [ ${#video_files[@]} -gt 0 ]; then
        # 创建raw_videos目录
        mkdir -p "$raw_videos_dir"
        
        echo -e "从数据集根目录发现${#video_files[@]}个视频文件，复制到raw_videos目录"
        
        # 原始文件到新文件的映射
        declare -A file_mapping
        
        for ((i=0; i<${#video_files[@]}; i++)); do
            file_path="${video_files[$i]}"
            # 生成简化的文件名，使用项目名称作为前缀
            ext="${file_path##*.}"
            safe_filename="${basename}$(printf "%02d" $((i+1))).${ext,,}"
            
            # 目标路径
            dest_path="$raw_videos_dir/$safe_filename"
            
            # 保存映射关系
            original_name=$(basename "$file_path")
            file_mapping["$original_name"]="$safe_filename"
            
            # 复制文件
            if [ ! -f "$dest_path" ]; then
                cp "$file_path" "$dest_path"
                echo "重命名视频: '$original_name' -> '$safe_filename'"
            fi
            
            # 删除原始视频文件，避免数据集重复
            if [ -f "$file_path" ] && [[ "$file_path" != "$raw_videos_dir"* ]]; then
                rm "$file_path"
                echo "删除原始视频文件: $file_path"
            fi
        done
        
        # 如果有多个文件被重命名，写入映射记录文件
        if [ ${#file_mapping[@]} -gt 1 ]; then
            mapping_file="$dataset_path/video_name_mapping.txt"
            {
                echo "原始文件名 -> 重命名后的文件名"
                echo "---------------------------------------------------"
                for orig in "${!file_mapping[@]}"; do
                    echo "$orig -> ${file_mapping[$orig]}"
                done
            } > "$mapping_file"
            echo "文件名映射记录已保存到: $mapping_file"
        fi
        
        # 更新raw_videos列表
        raw_videos=()
        while IFS= read -r -d '' file; do
            raw_videos+=("$file")
        done < <(find "$raw_videos_dir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) -print0)
    fi
    
    # 检查scenes目录是否存在且不为空
    scenes_empty=true
    if [ -d "$scenes_dir" ]; then
        scene_videos=()
        while IFS= read -r -d '' file; do
            scene_videos+=("$file")
        done < <(find "$scenes_dir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) -print0)
        
        if [ ${#scene_videos[@]} -gt 0 ]; then
            scenes_empty=false
        fi
    fi
    
    if [ "$split_scenes" = true ] && [ ${#raw_videos[@]} -gt 0 ] && ([ ! -d "$scenes_dir" ] || [ "$scenes_empty" = true ]); then
        # 存在原始视频但没有场景目录或场景目录为空，执行分场景
        mkdir -p "$scenes_dir"
        echo -e "找到${#raw_videos[@]}个原始视频文件，开始执行分场景..."
        
        # 分场景参数
        min_scene_length="3s"   # 使用3秒作为最短场景长度
        detector_type="content" # 内容检测器
        threshold="15"          # 之前是30，降低到只要差异超过15就分场景
        
        echo "使用更宽松的场景分割参数: 最短场景=$min_scene_length, 检测器=$detector_type, 阈值=$threshold"
        
        # 执行分场景命令
        for video_path in "${raw_videos[@]}"; do
            split_scene_cmd=(
                python "$SCRIPTS_DIR/split_scenes.py"
                "$video_path"
                "$scenes_dir"
                "--filter-shorter-than" "$min_scene_length"
                "--detector" "$detector_type"
                "--threshold" "$threshold"
            )
            run_command "${split_scene_cmd[@]}"
        done
        
        # 重新检查分场景结果
        scene_videos=()
        while IFS= read -r -d '' file; do
            scene_videos+=("$file")
        done < <(find "$scenes_dir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) -print0)
        
        if [ ${#scene_videos[@]} -gt 0 ]; then
            echo -e "${GREEN}分场景完成，生成了${#scene_videos[@]}个场景视频${NC}"
            
            # 在分场景完成后立即复制视频文件到caption目录
            caption_dir="$dataset_path/caption"
            mkdir -p "$caption_dir"
            
            # 复制scenes目录中的所有视频到caption目录
            copied_count=0
            for video_file in "${scene_videos[@]}"; do
                video_filename=$(basename "$video_file")
                caption_video_path="$caption_dir/$video_filename"
                
                # 如果目标文件不存在或者源文件比目标文件新，才复制
                if [ ! -f "$caption_video_path" ] || [ "$video_file" -nt "$caption_video_path" ]; then
                    cp "$video_file" "$caption_video_path"
                    ((copied_count++))
                    echo "复制视频到caption目录: $video_filename"
                fi
            done
            
            if [ $copied_count -gt 0 ]; then
                echo -e "${GREEN}已复制$copied_count个场景视频到caption目录${NC}"
            else
                echo "所有视频文件已存在于caption目录中，无需复制"
            fi
            
            # ======== 步骤3.1: 调整视频分辨率 ========
            echo -e "\n${BLUE}=== 步骤3.1: 调整视频分辨率 ===${NC}"
            
            # 清理分辨率字符串，提取数字部分
            clean_dims="$dims"
            if [[ "$clean_dims" == *"]"* ]]; then
                clean_dims=$(echo "$clean_dims" | sed 's/.*\] //')
            fi
            
            # 创建调整分辨率后的目录
            resized_dir="$dataset_path/caption_resized"
            mkdir -p "$resized_dir"
            
            # 检查是否已经有调整过大小的视频
            resized_videos=()
            while IFS= read -r -d '' file; do
                resized_videos+=("$file")
            done < <(find "$resized_dir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) -print0)
            
            if [ ${#resized_videos[@]} -gt 0 ] && [ ${#resized_videos[@]} -ge ${#scene_videos[@]} ]; then
                echo -e "发现${#resized_videos[@]}个调整过分辨率的视频文件，跳过分辨率调整步骤"
            else
                # 执行分辨率调整
                echo -e "开始调整视频分辨率至 ${CYAN}$clean_dims${NC}"
                
                resize_cmd=(
                    python "$SCRIPTS_DIR/resize_videos.py"
                    "$caption_dir"
                    "--output-dir" "$resized_dir"
                    "--target-size" "$clean_dims"
                    "--keep-aspect-ratio"
                )
                
                run_command "${resize_cmd[@]}"
                
                # 检查是否成功
                resized_videos=()
                while IFS= read -r -d '' file; do
                    resized_videos+=("$file")
                done < <(find "$resized_dir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) -print0)
                
                if [ ${#resized_videos[@]} -gt 0 ]; then
                    echo -e "${GREEN}分辨率调整完成，生成了${#resized_videos[@]}个调整后的视频${NC}"
                else
                    echo -e "${RED}分辨率调整失败，未生成视频文件${NC}"
                    echo -e "${YELLOW}将使用原始视频继续${NC}"
                    resized_dir="$caption_dir"
                fi
            fi
            
            # ======== 步骤3.2: 提取视频帧 ========
            echo -e "\n${BLUE}=== 步骤3.2: 提取视频帧 ===${NC}"
            
            # 创建帧提取目录
            frames_dir="$dataset_path/frames"
            mkdir -p "$frames_dir"
            
            # 检查是否已有提取的帧
            existing_frame_dirs=()
            while IFS= read -r -d '' dir; do
                existing_frame_dirs+=("$dir")
            done < <(find "$frames_dir" -mindepth 1 -maxdepth 1 -type d -print0)
            
            if [ ${#existing_frame_dirs[@]} -gt 0 ] && [ ${#existing_frame_dirs[@]} -ge ${#resized_videos[@]} ]; then
                echo -e "发现${#existing_frame_dirs[@]}个视频已提取帧，跳过帧提取步骤"
            else
                # 执行帧提取
                frames_count=$frames  # 直接使用提供的帧数
                echo -e "开始从每个视频提取${CYAN}$frames_count${NC}帧"
                
                extract_cmd=(
                    python "$SCRIPTS_DIR/extract_frames.py"
                    "$resized_dir"  # 从调整分辨率后的目录提取帧
                    "--output-dir" "$frames_dir"
                    "--num-frames" "$frames_count"
                    "--mode" "uniform"
                )
                
                run_command "${extract_cmd[@]}"
                
                # 检查是否成功
                existing_frame_dirs=()
                while IFS= read -r -d '' dir; do
                    existing_frame_dirs+=("$dir")
                done < <(find "$frames_dir" -mindepth 1 -maxdepth 1 -type d -print0)
                
                if [ ${#existing_frame_dirs[@]} -gt 0 ]; then
                    echo -e "${GREEN}帧提取完成，生成了${#existing_frame_dirs[@]}个视频帧目录${NC}"
                else
                    echo -e "${RED}帧提取失败，未生成帧目录${NC}"
                    echo -e "${YELLOW}将使用调整后的视频继续标注${NC}"
                fi
            fi
        else
            echo -e "${RED}分场景失败，未生成场景视频${NC}"
        fi
    else
        if [ ${#raw_videos[@]} -eq 0 ]; then
            echo -e "未找到原始视频文件，跳过分场景步骤"
        else
            echo -e "场景目录已存在且有视频文件，跳过分场景步骤"
        fi
        
        # 即使跳过分场景，也需要确保已存在的场景被复制到caption目录
        if [ -d "$scenes_dir" ]; then
            caption_dir="$dataset_path/caption"
            mkdir -p "$caption_dir"
            
            scene_videos=()
            while IFS= read -r -d '' file; do
                scene_videos+=("$file")
            done < <(find "$scenes_dir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) -print0)
            
            if [ ${#scene_videos[@]} -gt 0 ]; then
                copied_count=0
                for video_file in "${scene_videos[@]}"; do
                    video_filename=$(basename "$video_file")
                    caption_video_path="$caption_dir/$video_filename"
                    
                    if [ ! -f "$caption_video_path" ] || [ "$video_file" -nt "$caption_video_path" ]; then
                        cp "$video_file" "$caption_video_path"
                        ((copied_count++))
                        echo "复制已存在的视频到caption目录: $video_filename"
                    fi
                done
                
                if [ $copied_count -gt 0 ]; then
                    echo -e "${GREEN}已复制$copied_count个现有场景视频到caption目录${NC}"
                fi
            fi
        fi
    fi
    
    # ======== 步骤2: 检查标注文件 ========
    echo -e "\n${BLUE}=== 步骤2: 检查标注文件 ===${NC}"
    
    # 在数据集目录下创建caption目录
    caption_dir="$dataset_path/caption"
    mkdir -p "$caption_dir"
    
    # 标注文件应该在caption目录下
    caption_file="$caption_dir/caption.txt"
    caption_json="$caption_dir/captions.json"
    
    # 确保在预处理前有caption.txt文件
    has_caption_file=false
    has_caption_json=false
    
    if [ -f "$caption_file" ] && [ -s "$caption_file" ]; then
        has_caption_file=true
    fi
    
    if [ -f "$caption_json" ] && [ -s "$caption_json" ]; then
        has_caption_json=true
    fi
    
    # 优先使用现有的caption.txt文件，如果它存在
    if [ "$has_caption_file" = true ]; then
        echo -e "找到现有标注文件(caption.txt), 跳过标注步骤"
    # 如果有JSON文件但没有TXT文件，尝试转换
    elif [ "$has_caption_json" = true ]; then
        echo -e "找到JSON标注文件，正在转换为caption.txt格式..."
        
        # 尝试转换JSON到TXT
        if command -v python &> /dev/null; then
            # 使用Python进行JSON转换
            python -c "
import json, os
try:
    with open('$caption_json', 'r', encoding='utf-8') as f:
        captions_data = json.load(f)
    
    with open('$caption_file', 'w', encoding='utf-8') as f:
        if isinstance(captions_data, dict):
            for media_path, caption in captions_data.items():
                f.write(f'{media_path}|{caption}\\n')
        elif isinstance(captions_data, list):
            for item in captions_data:
                if isinstance(item, dict) and 'media_path' in item and 'caption' in item:
                    f.write(f'{item[\"media_path\"]}|{item[\"caption\"]}\\n')
    print('成功从captions.json创建caption.txt标注文件')
except Exception as e:
    print(f'转换标注文件时出错: {e}')
    exit(1)
"
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}成功将JSON标注转换为TXT格式${NC}"
                caption=false  # 设置为false跳过标注生成
            else
                echo -e "${RED}转换标注文件失败${NC}"
                # 继续尝试生成标注
            fi
        else
            echo -e "${YELLOW}警告: 未找到Python，无法转换JSON标注${NC}"
        fi
    fi
    
    # 再次检查caption.txt是否存在
    if [ -f "$caption_file" ] && [ -s "$caption_file" ]; then
        has_caption_file=true
    else
        has_caption_file=false
    fi
    
    # 如果仍然没有标注文件且用户要求标注，执行标注脚本
    if [ "$caption" = true ] && [ "$has_caption_file" = false ]; then
        # 执行标注命令
        echo -e "未找到标注文件，开始执行视频标注流程..."
        
        # 确定正确的输入目录 - 优先使用caption目录的视频
        # 首先检查caption目录中是否有视频文件
        caption_videos=()
        if [ -d "$caption_dir" ]; then
            while IFS= read -r -d '' file; do
                caption_videos+=("$file")
            done < <(find "$caption_dir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) -print0)
        fi
        
        # 然后检查scenes目录
        scenes_dir="$dataset_path/scenes"
        scene_videos=()
        if [ -d "$scenes_dir" ] && [ ${#caption_videos[@]} -eq 0 ]; then
            while IFS= read -r -d '' file; do
                scene_videos+=("$file")
            done < <(find "$scenes_dir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) -print0)
            
            # 如果有scenes视频但没有caption视频，自动复制
            if [ ${#scene_videos[@]} -gt 0 ] && [ ${#caption_videos[@]} -eq 0 ]; then
                echo "在caption目录中没有找到视频文件，将从scenes目录复制"
                copied_count=0
                for video_file in "${scene_videos[@]}"; do
                    video_filename=$(basename "$video_file")
                    caption_video_path="$caption_dir/$video_filename"
                    
                    cp "$video_file" "$caption_video_path"
                    ((copied_count++))
                    echo "复制视频到caption目录: $video_filename"
                done
                
                echo -e "${GREEN}已复制$copied_count个场景视频到caption目录${NC}"
                # 更新caption_videos列表
                while IFS= read -r -d '' file; do
                    caption_videos+=("$file")
                done < <(find "$caption_dir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) -print0)
            fi
        fi
        
        # 选择适当的输入目录
        input_dir=""
        if [ ${#caption_videos[@]} -gt 0 ]; then
            input_dir="$caption_dir"
            echo -e "将使用caption目录中的视频文件进行标注 (${#caption_videos[@]}个视频)"
        elif [ ${#scene_videos[@]} -gt 0 ]; then
            input_dir="$scenes_dir"
            echo -e "${YELLOW}警告: 使用scenes目录中的视频文件进行标注，未找到调整分辨率后的视频${NC}"
        else
            input_dir="$dataset_path"
            echo -e "${YELLOW}警告: 使用数据集根目录进行标注，未找到任何可用视频${NC}"
        fi
            
        echo -e "将使用${CYAN}$input_dir${NC}目录中的视频文件进行标注"
        
        # 准备标注输出路径
        output_json="$dataset_path/captions.json"
        
        # 在执行标注命令前，先检查目录中是否确实有视频文件
        video_files_in_input=()
        while IFS= read -r -d '' file; do
            video_files_in_input+=("$file")
        done < <(find "$input_dir" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) -print0)
            
        if [ ${#video_files_in_input[@]} -eq 0 ]; then
            echo -e "${RED}错误: 未在$input_dir目录中找到视频文件。请先添加视频文件或选择正确的数据集目录。${NC}"
            return 1
        fi
        
        echo -e "在${CYAN}$input_dir${NC}目录中找到${#video_files_in_input[@]}个视频文件"
        
        # 根据用户选择的标注方式决定使用哪个标注脚本
        use_moonshot=false
        if [[ "$caption_method" == *"Moonshot API"* ]]; then
            use_moonshot=true
        fi
        
        if [ "$use_moonshot" = true ]; then
            # 使用Moonshot API进行标注
            echo -e "使用${CYAN}Moonshot API${NC}进行视频标注..."
            
            # 检查API密钥文件是否存在
            api_key_file="$PROJECT_DIR/api_key.txt"
            if [ ! -f "$api_key_file" ]; then
                # 如果API密钥文件不存在，创建一个提示用户输入的文件
                {
                    echo "# 请将您的Moonshot API密钥粘贴在下面一行，然后保存文件"
                    echo "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                    echo "# 请到 [https://platform.moonshot.cn/docs/pricing/chat](https://platform.moonshot.cn/docs/pricing/chat) 申请API密钥"
                } > "$api_key_file"
                echo -e "${RED}未找到API密钥文件。已创建模板文件: $api_key_file${NC}"
                echo -e "${RED}请在此文件中输入您的Moonshot API密钥并再次运行标注。${NC}"
                return 1
            fi
            
            # 使用Moonshot API标注脚本
            caption_cmd=(
                python "$SCRIPTS_DIR/caption_with_moonshot.py"
                "$input_dir"  # 使用选定的输入目录
                "--output" "$output_json"
            )
        else
            # 使用本地LLaVA模型进行标注
            echo -e "使用${CYAN}本地LLaVA模型${NC}进行视频标注..."
                
            # 检查是否有CUDA支持
            if ! python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
                echo -e "${YELLOW}警告: 检测到您的PyTorch没有CUDA支持，本地模型标注可能非常缓慢。建议安装CUDA支持的PyTorch或切换到Moonshot API标注。${NC}"
            fi
            
            # 检查本地模型是否存在
            if [ ! -d "$LLAVA_MODEL_PATH" ]; then
                echo -e "${RED}错误: LLaVA-NeXT-Video-7B-hf模型未下载。${NC}"
                echo -e "${YELLOW}请先运行 'python download_llava_model.py' 下载模型，或选择使用Moonshot API${NC}"
                
                echo -e "是否切换到Moonshot API继续? [Y/n]"
                read -r switch_to_moonshot
                
                if [[ ! "$switch_to_moonshot" =~ ^[Nn]$ ]]; then
                    use_moonshot=true
                    echo -e "${YELLOW}切换到Moonshot API进行标注${NC}"
                    
                    # 检查API密钥文件是否存在
                    api_key_file="$PROJECT_DIR/api_key.txt"
                    if [ ! -f "$api_key_file" ]; then
                        # 如果API密钥文件不存在，创建一个提示用户输入的文件
                        {
                            echo "# 请将您的Moonshot API密钥粘贴在下面一行，然后保存文件"
                            echo "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                            echo "# 请到 [https://platform.moonshot.cn/docs/pricing/chat](https://platform.moonshot.cn/docs/pricing/chat) 申请API密钥"
                        } > "$api_key_file"
                        echo -e "${RED}未找到API密钥文件。已创建模板文件: $api_key_file${NC}"
                        echo -e "${RED}请在此文件中输入您的Moonshot API密钥并再次运行标注。${NC}"
                        return 1
                    fi
                    
                    # 使用Moonshot API标注脚本
                    caption_cmd=(
                        python "$SCRIPTS_DIR/caption_with_moonshot.py"
                        "$input_dir"  # 使用选定的输入目录
                        "--output" "$output_json"
                    )
                    return
                else
                    echo -e "${RED}无法继续，请先下载LLaVA模型${NC}"
                    return 1
                fi
            fi
            
            # 暂时禁用离线模式以使用本地模型
            export TRANSFORMERS_OFFLINE=0
            export HF_HUB_OFFLINE=0
            export HF_DATASETS_OFFLINE=0
            
            # 使用本地LLaVA模型标注脚本
            caption_cmd=(
                python "$SCRIPTS_DIR/caption_videos.py"
                "$input_dir"  # 使用选定的输入目录
                "--output" "$output_json"
                "--captioner-type" "llava_next_7b"  # 使用LLaVA-NeXT-7B模型
                "--model-path" "$LLAVA_MODEL_PATH"  # 指定本地模型路径
            )
            
            # 标注完成后恢复离线模式
            trap 'export TRANSFORMERS_OFFLINE=1; export HF_HUB_OFFLINE=1; export HF_DATASETS_OFFLINE=1' EXIT
        fi
        
        # 根据选择的标注方法显示相应的命令日志
        method_name="Moonshot API"
        if [ "$use_moonshot" = false ]; then
            method_name="本地LLaVA模型"
        fi
        echo -e "执行 ${CYAN}$method_name${NC} 标注命令: ${caption_cmd[*]}"
        run_command "${caption_cmd[@]}"
        
        # 检查标注是否成功
        caption_file_exists=false
        caption_json_exists=false
        output_json_exists=false
        
        if [ -f "$caption_file" ] && [ -s "$caption_file" ]; then
            caption_file_exists=true
        fi
        
        if [ -f "$caption_json" ] && [ -s "$caption_json" ]; then
            caption_json_exists=true
        fi
        
        if [ -f "$output_json" ] && [ -s "$output_json" ]; then
            output_json_exists=true
        fi
        
        # 如果标注成功，继续处理流程
        if [ "$caption_file_exists" = true ] || [ "$caption_json_exists" = true ] || [ "$output_json_exists" = true ]; then
            echo -e "${GREEN}标注已生成，继续处理流程${NC}"
                
            # 如果标注文件生成在数据集根目录，复制到caption目录
            if [ "$output_json_exists" = true ]; then
                echo -e "复制标注JSON文件到caption目录: $caption_json"
                cp "$output_json" "$caption_json"
            fi
        else
            echo -e "${RED}标注视频失败${NC}"
            return 1
        fi
        
        # 如果生成了JSON文件但没有TXT文件，转换JSON为TXT
        # 检查所有可能的JSON标注文件位置
        json_files=(
            "$caption_json"  # caption目录中的JSON文件
            "$dataset_path/captions.json"  # 数据集根目录中的JSON文件
            "$scenes_dir/captions.json"  # scenes目录中的JSON文件
        )
        
        # 找到最新的有效JSON标注文件
        valid_json_file=""
        newest_time=0
        
        for json_file in "${json_files[@]}"; do
            if [ -f "$json_file" ] && [ -s "$json_file" ]; then
                file_time=$(stat -c %Y "$json_file")
                if [ -z "$valid_json_file" ] || [ "$file_time" -gt "$newest_time" ]; then
                    valid_json_file="$json_file"
                    newest_time="$file_time"
                fi
            fi
        done
        
        # 如果有效的JSON文件与caption目录中的不同，复制到caption目录
        if [ -n "$valid_json_file" ] && [ "$valid_json_file" != "$caption_json" ]; then
            echo -e "复制最新的标注JSON文件到caption目录: $valid_json_file -> $caption_json"
            cp "$valid_json_file" "$caption_json"
        fi
        
        # 试图将JSON标注转换为TXT格式（如果存在JSON文件但没有TXT文件）
        if [ -f "$caption_json" ] && [ ! -f "$caption_file" -o ! -s "$caption_file" ]; then
            echo -e "将JSON标注转换为TXT格式..."
            
            # 使用Python进行转换
            if command -v python &> /dev/null; then
                python -c "
import json, os
try:
    with open('$caption_json', 'r', encoding='utf-8') as f:
        captions_data = json.load(f)
    
    with open('$caption_file', 'w', encoding='utf-8') as f:
        if isinstance(captions_data, dict):
            for media_path, caption in captions_data.items():
                # 获取基本文件名
                base_filename = os.path.basename(media_path)
                f.write(f'{base_filename}|{caption}\\n')
        elif isinstance(captions_data, list):
            for item in captions_data:
                if isinstance(item, dict) and 'caption' in item and 'media_path' in item:
                    # 只使用文件名作为关键字
                    media_name = os.path.basename(item['media_path'])
                    f.write(f'{media_name}|{item[\"caption\"]}\\n')
    print('成功将JSON标注转换为TXT格式')
except Exception as e:
    print(f'转换JSON标注到TXT格式时出错: {str(e)}')
    exit(1)
"
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}成功将JSON标注转换为TXT格式: $caption_file${NC}"
                else
                    echo -e "${RED}转换JSON标注到TXT格式时出错${NC}"
                fi
            else
                echo -e "${YELLOW}警告: 未找到Python，无法转换JSON标注${NC}"
            fi
        fi
    else
        echo -e "找到现有标注文件，跳过标注步骤"
    fi
    
    # 自动向标注文件添加触发词
    if [ "$add_trigger" = true ]; then
        # 提取触发词
        trigger_word=$(extract_trigger_word "$basename")
        echo -e "为标注文件添加触发词 '${CYAN}$trigger_word${NC}'"
        
        # 添加触发词到标注文件
        trigger_word_script="$SCRIPTS_DIR/add_trigger_word_to_captions.py"
        if [ -f "$trigger_word_script" ]; then
            # 调用触发词添加脚本
            trigger_cmd=(
                python "$trigger_word_script"
                "$dataset_path"
                "--trigger-word" "$trigger_word"
            )
            echo -e "执行触发词添加命令: ${trigger_cmd[*]}"
            run_command "${trigger_cmd[@]}"
            echo -e "${GREEN}触发词添加完成${NC}"
        else
            echo -e "${YELLOW}警告: 触发词添加脚本不存在: $trigger_word_script${NC}"
            
            # 如果没有脚本，使用简单的sed进行替换
            if [ -f "$caption_file" ]; then
                # 备份原文件
                cp "$caption_file" "${caption_file}.bak"
                
                # 检查触发词格式并添加
                if ! grep -q "<$trigger_word>" "$caption_file"; then
                    # 在每行的|后添加触发词
                    sed -i "s/|/<$trigger_word> /" "$caption_file"
                    echo -e "${GREEN}使用sed添加触发词 <$trigger_word> 到标注文件${NC}"
                else
                    echo -e "标注文件中已包含触发词 <$trigger_word>"
                fi
            fi
        fi
    fi
    
    # 检查标注文件是否存在
    if [ ! -f "$caption_file" ] || [ ! -s "$caption_file" ]; then
        echo -e "${RED}错误: 标注文件(caption.txt)不存在或为空，无法进行预处理${NC}"
        return 1
    fi
    
    # ======== 步骤3: 数据预处理部分 ========
    echo -e "\n${BLUE}=== 步骤3: 开始数据预处理 ===${NC}"
    
    # 初始化预处理路径
    precomputed_path="$dataset_path/.precomputed"
    
    # 创建预处理目录（如果不存在）
    if [ ! -d "$precomputed_path" ]; then
        echo -e "创建预处理目录: $precomputed_path"
        mkdir -p "$precomputed_path"
    fi

    # 检查是否已有预处理数据
    latents_dir="$precomputed_path/latents"
    conditions_dir="$precomputed_path/conditions"
    
    has_precomputed_data=false
    
    # 更严格的检查: 两个目录必须存在且内部必须有文件
    if [ -d "$latents_dir" ] && [ -d "$conditions_dir" ]; then
        latents_files=($(ls -A "$latents_dir" 2>/dev/null))
        conditions_files=($(ls -A "$conditions_dir" 2>/dev/null))
        
        # 只有当两个目录都非空时才认为有效
        if [ ${#latents_files[@]} -gt 0 ] && [ ${#conditions_files[@]} -gt 0 ]; then
            has_precomputed_data=true
            echo -e "${GREEN}找到已预处理的数据: ${#latents_files[@]} 个潜在文件, ${#conditions_files[@]} 个条件文件${NC}"
        else
            echo -e "${YELLOW}警告: 预处理目录存在但为空: 潜在文件=${#latents_files[@]}, 条件文件=${#conditions_files[@]}${NC}"
        fi
    else
        echo -e "预处理目录不存在或不完整: latents_dir=$([ -d "$latents_dir" ] && echo "存在" || echo "不存在"), conditions_dir=$([ -d "$conditions_dir" ] && echo "存在" || echo "不存在")"
    fi
    
    # 决定是否需要运行预处理
    if [ "$preprocess" = true ] && [ "$has_precomputed_data" = true ]; then
        echo -e "${GREEN}已找到预处理数据，跳过预处理步骤${NC}"
    elif [ "$preprocess" = true ]; then  # 需要预处理且没有现有预处理数据
        echo -e "检测到预处理数据不存在，开始执行预处理..."
        
        # 修复视频与标题不匹配问题
        fix_mismatch_script="$SCRIPTS_DIR/auto_fix_video_caption_mismatch.py"
        if [ -f "$fix_mismatch_script" ]; then
            echo -e "在预处理前检查并修复视频与标题不匹配问题..."
            
            # 执行修复脚本
            fix_cmd=(
                python "$fix_mismatch_script"
                "$dataset_path"
            )
            run_command "${fix_cmd[@]}"
        else
            echo -e "${YELLOW}警告: 自动修复脚本不存在: $fix_mismatch_script${NC}"
        fi
            
        # 确保预处理目录已创建
        mkdir -p "$precomputed_path/latents"
        mkdir -p "$precomputed_path/conditions"
        
        # 检查caption.txt内容
        if [ -f "$caption_file" ]; then
            echo -e "标注文件内容预览:"
            head -n 5 "$caption_file"
            echo "..."
        fi

        # 定义预处理命令并执行
        # 首先检查必要的预处理文件是否存在
        dataset_script="$SCRIPTS_DIR/preprocess_dataset.py"
        if [ ! -f "$dataset_script" ]; then
            echo -e "${RED}错误: 预处理脚本不存在: $dataset_script${NC}"
            return 1
        fi
                
        # 复制标注文件到scenes目录
        if [ -d "$scenes_dir" ]; then
            scenes_caption_file="$scenes_dir/caption.txt"
            if [ ! -f "$scenes_caption_file" ] && [ -f "$caption_file" ]; then
                cp "$caption_file" "$scenes_caption_file"
                echo -e "复制标注文件到scenes目录: $scenes_caption_file"
            fi
        fi
        
        # 修改JSON文件中的路径格式，仅使用文件名而非路径
        source_json_path="$dataset_path/captions.json"
        
        if [ -f "$source_json_path" ]; then
            # 使用Python修改JSON
            if command -v python &> /dev/null; then
                python -c "
import json, os
try:
    with open('$source_json_path', 'r', encoding='utf-8') as f:
        captions_data = json.load(f)
    
    # 将media_path从相对路径改为纯文件名
    if isinstance(captions_data, list):
        for item in captions_data:
            if 'media_path' in item:
                # 提取文件名
                item['media_path'] = os.path.basename(item['media_path'])
    elif isinstance(captions_data, dict):
        modified_data = {}
        for key, value in captions_data.items():
            # 使用文件名作为新的键
            new_key = os.path.basename(key)
            modified_data[new_key] = value
        captions_data = modified_data
    
    # 写回修改后的JSON文件
    with open('$source_json_path', 'w', encoding='utf-8') as f:
        json.dump(captions_data, f, ensure_ascii=False, indent=2)
    
    print('成功修改JSON文件中的路径格式')
except Exception as e:
    print(f'修改JSON文件失败: {str(e)}')
    exit(1)
"
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}成功修改JSON文件中的路径格式: $source_json_path${NC}"
                else
                    echo -e "${RED}修改JSON文件失败${NC}"
                fi
            else
                echo -e "${YELLOW}警告: 未找到Python，无法修改JSON文件${NC}"
            fi
        else
            echo -e "${YELLOW}警告: 找不到JSON文件: $source_json_path${NC}"
        fi
                
        # 复制标注JSON文件到caption目录
        caption_json_path="$caption_dir/captions.json"
        if [ -f "$source_json_path" ]; then
            cp "$source_json_path" "$caption_json_path"
            echo -e "复制修改后的标注JSON文件到caption目录: $caption_json_path"
        fi
        
        # 第一次将标注文件复制到数据集根目录
        root_caption_path="$dataset_path/caption.txt"
        if [ -f "$caption_file" ]; then
            cp "$caption_file" "$root_caption_path"
            echo -e "复制标注文件到数据集根目录: $root_caption_path"
        fi
                
        # 创建media_path.txt文件，预处理脚本需要这个文件来找到视频文件
        # 首先找到视频文件
        video_filenames=()
        for dir in "$dataset_path" "$caption_dir" "$scenes_dir"; do
            if [ -d "$dir" ]; then
                while IFS= read -r -d '' file; do
                    video_filenames+=("$(basename "$file")")
                done < <(find "$dir" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) -print0)
            fi
        done
        
        # 删除重复项
        unique_video_filenames=($(echo "${video_filenames[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))
        
        # 创建media_path.txt文件
        media_path_txt="$dataset_path/media_path.txt"
        > "$media_path_txt"  # 清空文件
        for filename in "${unique_video_filenames[@]}"; do
            echo "$filename" >> "$media_path_txt"
        done
        echo -e "已创建media_path.txt文件，包含${#unique_video_filenames[@]}个视频文件路径"
        
        # 使用预处理脚本
        preprocess_script=""
        
        # 检查不同的预处理脚本
        zero_workers_script="$SCRIPTS_DIR/preprocess_zero_workers.py"
        fix_resolution_script="$SCRIPTS_DIR/fix_resolution_wrapper.py"
        preprocess_wrapper="$SCRIPTS_DIR/preprocess_wrapper.py"
        
        # 优先使用零工作线程脚本
        if [ -f "$zero_workers_script" ]; then
            preprocess_script="$zero_workers_script"
            echo -e "使用零工作线程修复脚本: $zero_workers_script"
        elif [ -f "$fix_resolution_script" ]; then
            preprocess_script="$fix_resolution_script"
            echo -e "使用分辨率格式修复包装器: $fix_resolution_script"
        elif [ -f "$preprocess_wrapper" ]; then
            preprocess_script="$preprocess_wrapper"
            echo -e "使用预处理包装器: $preprocess_wrapper"
        else
            preprocess_script="$dataset_script"
            echo -e "使用原始预处理脚本: $dataset_script"
        fi
        
        # 设置分辨率桶
        resolution_bucket="$dimensions"
        echo -e "使用分辨率: ${CYAN}$resolution_bucket${NC}, 帧数: ${CYAN}$frames${NC}"
        
        # 检测可用设备
        device="cpu"
        if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
            device="cuda"
            gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
            gpu_mem=$(python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / (1024**3))")
            echo -e "使用GPU: ${CYAN}$gpu_name${NC}, 显存: ${CYAN}${gpu_mem}GB${NC}"
        else
            echo -e "使用CPU进行预处理"
        fi
        
        # 构建预处理命令
        preprocess_cmd=(
            python "$preprocess_script"
            "$dataset_path"  # 输入数据集目录
            "--resolution-buckets" "$resolution_bucket"
            "--output-dir" "$precomputed_path"
            "--batch-size" "1"  # 小批量大小防止内存问题
            "--num-workers" "0"  # 强制使用单线程，防止序列化错误
            "--device" "$device"  # 动态使用可用的最佳设备
            "--frames" "$frames"  # 显式传递帧数参数
        )
        
        # 执行预处理命令
        echo -e "执行预处理命令: ${preprocess_cmd[*]}"
        run_command "${preprocess_cmd[@]}"
        
        # 检查预处理是否成功
        if [ -d "$latents_dir" ] && [ -d "$conditions_dir" ]; then
            latents_files=($(ls -A "$latents_dir" 2>/dev/null))
            conditions_files=($(ls -A "$conditions_dir" 2>/dev/null))
            
            if [ ${#latents_files[@]} -gt 0 ] && [ ${#conditions_files[@]} -gt 0 ]; then
                echo -e "${GREEN}预处理成功完成，生成了${#latents_files[@]}个潜在文件和${#conditions_files[@]}个条件文件${NC}"
                has_precomputed_data=true
            else
                echo -e "${RED}预处理失败，未生成所需文件${NC}"
                has_precomputed_data=false
                return 1
            fi
        else
            echo -e "${RED}预处理失败，未创建所需目录${NC}"
            has_precomputed_data=false
            return 1
        fi
    elif [ "$preprocess" = false ]; then
        echo -e "${YELLOW}跳过预处理步骤（用户选择）${NC}"
    fi
    
    # 如果用户只想预处理数据，到此结束
    if [ "$only_preprocess" = true ]; then
        echo -e "${GREEN}预处理步骤已完成，根据用户选择跳过训练${NC}"
        return 0
    fi
    
    # 如果没有配置模板名称，跳过训练
    if [ -z "$config_name" ]; then
        echo -e "${YELLOW}没有指定配置模板名称，跳过训练步骤${NC}"
        return 0
    fi
    
    # ======== 步骤4: 创建临时配置文件并进行训练 ========
    echo -e "\n${BLUE}=== 步骤4: 创建临时配置文件并进行训练 ===${NC}"
    
    # 查找匹配的配置模板
    config_template=""
    for i in "${!CONFIG_NAMES[@]}"; do
        if [ "${CONFIG_NAMES[$i]}" = "$config_name" ]; then
            config_template="${CONFIG_FILES[$i]}"
            break
        fi
    done
    
    if [ -z "$config_template" ]; then
        echo -e "${RED}错误: 找不到配置 $config_name${NC}"
        return 1
    fi
    
    # 创建临时配置目录
    temp_config_dir="$PROJECT_DIR/temp_configs"
    mkdir -p "$temp_config_dir"
    
    # 创建临时配置文件
    temp_config_path="$temp_config_dir/${basename}_temp_config.yaml"
    
    echo -e "创建临时配置文件: ${CYAN}$temp_config_path${NC}"
    
    # 读取模板配置文件
    if [ -f "$config_template" ]; then
        # 使用Python处理YAML文件
        if command -v python &> /dev/null; then
            python -c "
import yaml, os
try:
    # 读取模板配置
    with open('$config_template', 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # 修改配置项目
    # 更新preprocessed_data_root
    if 'data' not in config_data:
        config_data['data'] = {}
    config_data['data']['preprocessed_data_root'] = '$precomputed_path'
    
    # 更新lora rank
    if 'lora' not in config_data:
        config_data['lora'] = {}
    config_data['lora']['rank'] = $rank
    config_data['lora']['alpha'] = $rank
    
    # 更新video_dims
    if 'validation' not in config_data:
        config_data['validation'] = {}
    dims = '$dimensions'.split('x')
    frames_val = int('$frames')
    if len(dims) == 2:
        config_data['validation']['video_dims'] = [int(dims[0]), int(dims[1]), frames_val]
    
    # 更新输出目录
    config_data['output_dir'] = f'outputs/{basename}_lora_r{$rank}_int8-quanto'
    
    # 保存修改后的配置
    with open('$temp_config_path', 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    print('成功创建临时配置文件')
except Exception as e:
    print(f'创建临时配置文件出错: {str(e)}')
    exit(1)
"
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}成功创建临时配置文件: $temp_config_path${NC}"
            else
                echo -e "${RED}创建临时配置文件失败${NC}"
                return 1
            fi
        else
            # 如果没有Python，尝试使用sed/awk手动修改
            cp "$config_template" "$temp_config_path"
            echo -e "${YELLOW}警告: 未找到Python，使用简单复制创建配置文件${NC}"
        fi
    else
        echo -e "${RED}错误: 配置模板文件不存在: $config_template${NC}"
        return 1
    fi
    
    # 显示创建的配置文件内容
    echo -e "\n临时配置文件内容预览:"
    head -n 20 "$temp_config_path"
    echo "..."
    
    # 执行训练
    echo -e "\n${BLUE}开始训练...${NC}"
    
    train_cmd=(
        python "$SCRIPTS_DIR/train.py"
        "$temp_config_path"
    )
    
    echo -e "执行训练命令: ${train_cmd[*]}"
    run_command "${train_cmd[@]}"
    
    # 检查训练是否成功
    output_dir="$PROJECT_DIR/outputs/${basename}_lora_r${rank}_int8-quanto"
    if [ -d "$output_dir" ]; then
        checkpoint_files=()
        while IFS= read -r -d '' file; do
            checkpoint_files+=("$file")
        done < <(find "$output_dir" -name "*.safetensors" -print0)
        
        if [ ${#checkpoint_files[@]} -gt 0 ]; then
            echo -e "${GREEN}训练成功完成，生成了${#checkpoint_files[@]}个检查点文件${NC}"
            echo -e "检查点文件保存在: ${CYAN}$output_dir${NC}"
            return 0
        else
            echo -e "${YELLOW}训练可能没有完全成功，未找到.safetensors检查点文件${NC}"
            return 1
        fi
    else
        echo -e "${RED}训练失败，未找到输出目录: $output_dir${NC}"
        return 1
    fi
}

# 主函数
function main() {
    show_title
    
    # 读取配置文件
    load_config_files
    
    # 1. 输入项目名称
    echo -e "${BLUE}步骤1: 输入项目名称${NC}"
    echo -e "请输入项目名称（例如: character01, scene01）:"
    read -r basename
    
    if [ -z "$basename" ]; then
        echo -e "${RED}错误: 项目名称不能为空${NC}"
        exit 1
    fi
    
    # 2. 选择分辨率
    echo -e "\n${BLUE}步骤2: 选择分辨率${NC}"
    echo -e "可用的分辨率:"
    for i in "${!RESOLUTIONS_DIMS[@]}"; do
        echo -e "  $i) ${RESOLUTIONS_DIMS[$i]}"
    done
    
    echo -e "请选择分辨率 (输入编号):"
    read -r dims_index
    
    if [[ ! "$dims_index" =~ ^[0-9]+$ ]] || [ "$dims_index" -lt 0 ] || [ "$dims_index" -ge ${#RESOLUTIONS_DIMS[@]} ]; then
        echo -e "${RED}错误: 无效的分辨率选择，使用默认分辨率 ${RESOLUTIONS_DIMS[3]}${NC}"
        dims_index=3  # 默认使用512x512
    fi
    
    dims="${RESOLUTIONS_DIMS[$dims_index]}"
    echo -e "选择的分辨率: ${CYAN}$dims${NC}"
    
    # 3. 选择帧数
    echo -e "\n${BLUE}步骤3: 选择帧数${NC}"
    echo -e "可用的帧数:"
    for i in "${!FRAME_COUNTS[@]}"; do
        echo -e "  $i) ${FRAME_COUNTS[$i]}"
    done
    
    echo -e "请选择帧数 (输入编号):"
    read -r frames_index
    
    if [[ ! "$frames_index" =~ ^[0-9]+$ ]] || [ "$frames_index" -lt 0 ] || [ "$frames_index" -ge ${#FRAME_COUNTS[@]} ]; then
        echo -e "${RED}错误: 无效的帧数选择，使用默认帧数 ${FRAME_COUNTS[0]}${NC}"
        frames_index=0  # 默认使用25帧
    fi
    
    frames="${FRAME_COUNTS[$frames_index]}"
    echo -e "选择的帧数: ${CYAN}$frames${NC}"
    
    # 4. 选择是否分场景
    echo -e "\n${BLUE}步骤4: 是否需要分场景${NC}"
    echo -e "是否需要将长视频拆分成场景? [Y/n]"
    read -r split_scenes_input
    
    split_scenes=true
    if [[ "$split_scenes_input" =~ ^[Nn]$ ]]; then
        split_scenes=false
        echo -e "不进行场景拆分"
    else
        echo -e "将进行场景拆分"
    fi
    
    # 5. 选择是否自动标注
    echo -e "\n${BLUE}步骤5: 是否需要自动标注${NC}"
    echo -e "是否需要自动标注视频? [Y/n]"
    read -r caption_input
    
    caption=true
    if [[ "$caption_input" =~ ^[Nn]$ ]]; then
        caption=false
        echo -e "不进行自动标注"
    else
        echo -e "将进行自动标注"
    fi
    
    # 6. 选择标注方法
    caption_method="本地LLaVA模型"
    if [ "$caption" = true ]; then
        echo -e "\n${BLUE}标注方法选择${NC}"
        echo -e "  1) 本地LLaVA模型 (默认，无需API密钥)"
        echo -e "  2) Moonshot API (推荐，需要API密钥)"
        
        echo -e "请选择标注方法 (输入编号):"
        read -r caption_method_index
        
        if [ "$caption_method_index" = "2" ]; then
            caption_method="Moonshot API (推荐，需要API密钥)"
            echo -e "选择了 ${CYAN}Moonshot API${NC} 标注方法"
        else
            echo -e "选择了 ${CYAN}本地LLaVA模型${NC} 标注方法"
        fi
    fi
    
    # 7. 选择是否预处理
    echo -e "\n${BLUE}步骤7: 是否需要预处理${NC}"
    echo -e "是否需要进行数据预处理? [Y/n]"
    read -r preprocess_input
    
    preprocess=true
    if [[ "$preprocess_input" =~ ^[Nn]$ ]]; then
        preprocess=false
        echo -e "不进行数据预处理"
    else
        echo -e "将进行数据预处理"
    fi
    
    # 8. 选择是否只进行预处理
    echo -e "\n${BLUE}步骤8: 是否只预处理不训练${NC}"
    echo -e "是否只进行预处理，不进行训练? [y/N]"
    read -r only_preprocess_input
    
    only_preprocess=false
    if [[ "$only_preprocess_input" =~ ^[Yy]$ ]]; then
        only_preprocess=true
        echo -e "只进行预处理，不进行训练"
    else
        echo -e "将进行预处理和训练"
    fi
    
    # 9. 选择配置模板（如果要训练）
    config_name=""
    rank=64  # 默认LoRA rank
    
    if [ "$only_preprocess" = false ]; then
        echo -e "\n${BLUE}步骤9: 选择配置模板${NC}"
        echo -e "可用的配置模板:"
        for i in "${!CONFIG_NAMES[@]}"; do
            echo -e "  $i) ${CONFIG_NAMES[$i]}"
        done
        
        echo -e "请选择配置模板 (输入编号):"
        read -r config_index
        
        if [[ ! "$config_index" =~ ^[0-9]+$ ]] || [ "$config_index" -lt 0 ] || [ "$config_index" -ge ${#CONFIG_NAMES[@]} ]; then
            echo -e "${RED}错误: 无效的配置模板选择${NC}"
            exit 1
        fi
        
        config_name="${CONFIG_NAMES[$config_index]}"
        echo -e "选择的配置模板: ${CYAN}$config_name${NC}"
        
        # 10. 设置LoRA秩
        echo -e "\n${BLUE}步骤10: 设置LoRA秩${NC}"
        echo -e "请设置LoRA秩 (推荐: 32-64):"
        read -r rank_input
        
        if [[ "$rank_input" =~ ^[0-9]+$ ]] && [ "$rank_input" -gt 0 ]; then
            rank="$rank_input"
        fi
        
        echo -e "使用LoRA秩: ${CYAN}$rank${NC}"
    fi
    
    # 11. 添加触发词
    echo -e "\n${BLUE}步骤11: 是否添加触发词${NC}"
    echo -e "是否自动添加触发词到标注? [Y/n]"
    read -r add_trigger_input
    
    add_trigger=true
    if [[ "$add_trigger_input" =~ ^[Nn]$ ]]; then
        add_trigger=false
        echo -e "不添加触发词"
    else
        echo -e "将添加触发词 <${CYAN}$(extract_trigger_word "$basename")${NC}> 到标注"
    fi
    
    # 显示所有选择的参数
    echo -e "\n${BLUE}=== 任务配置摘要 ===${NC}"
    echo -e "项目名称: ${CYAN}$basename${NC}"
    echo -e "分辨率: ${CYAN}$dims${NC}"
    echo -e "帧数: ${CYAN}$frames${NC}"
    echo -e "分场景: $([ "$split_scenes" = true ] && echo "${GREEN}是${NC}" || echo "${YELLOW}否${NC}")"
    echo -e "自动标注: $([ "$caption" = true ] && echo "${GREEN}是${NC} (${CYAN}$caption_method${NC})" || echo "${YELLOW}否${NC}")"
    echo -e "数据预处理: $([ "$preprocess" = true ] && echo "${GREEN}是${NC}" || echo "${YELLOW}否${NC}")"
    echo -e "只预处理不训练: $([ "$only_preprocess" = true ] && echo "${GREEN}是${NC}" || echo "${YELLOW}否${NC}")"
    
    if [ "$only_preprocess" = false ]; then
        echo -e "配置模板: ${CYAN}$config_name${NC}"
        echo -e "LoRA秩: ${CYAN}$rank${NC}"
    fi
    
    echo -e "添加触发词: $([ "$add_trigger" = true ] && echo "${GREEN}是${NC} (${CYAN}$(extract_trigger_word "$basename")${NC})" || echo "${YELLOW}否${NC}")"
    
    # 最终确认
    echo -e "\n${BLUE}是否开始执行训练流程? [Y/n]${NC}"
    read -r confirm
    
    if [[ "$confirm" =~ ^[Nn]$ ]]; then
        echo -e "已取消执行"
        exit 0
    fi
    
    # 执行训练流程
    echo -e "\n${BLUE}开始执行训练流程...${NC}"
    
    run_pipeline "$basename" "$dims" "$frames" "$config_name" "$rank" "$split_scenes" "$caption" "$preprocess" "$only_preprocess" "$add_trigger" "$caption_method"
    
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "\n${GREEN}训练流程成功完成!${NC}"
    else
        echo -e "\n${RED}训练流程失败，返回代码: $exit_code${NC}"
    fi
}

# 执行主函数
main
