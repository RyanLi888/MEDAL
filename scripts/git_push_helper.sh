#!/bin/bash
# ============================================================
# Git推送辅助脚本
# ============================================================
# 解决Git推送时的身份验证问题
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Git推送辅助工具"
echo "=========================================="
echo ""

# 检查是否有未提交的更改
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  检测到未提交的更改，请先提交："
    git status --short
    echo ""
    read -p "是否现在提交？(y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add -A
        git commit -m "Auto commit: $(date +'%Y-%m-%d %H:%M:%S')"
    else
        echo "❌ 已取消"
        exit 1
    fi
fi

# 检查是否有待推送的提交
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u} 2>/dev/null || echo "")
BASE=$(git merge-base @ @{u} 2>/dev/null || echo "")

if [ -z "$REMOTE" ]; then
    echo "⚠️  未设置上游分支，尝试推送..."
    git push -u origin main
    exit 0
fi

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "✓ 本地和远程已同步，无需推送"
    exit 0
fi

echo "📤 准备推送提交..."
echo ""

# 尝试推送
echo "方法1: 使用HTTPS（需要Personal Access Token）"
echo "----------------------------------------"
echo "如果使用HTTPS，您需要："
echo "1. 在GitHub创建Personal Access Token (PAT)"
echo "2. 使用token作为密码推送"
echo ""
echo "或者使用以下命令配置凭据："
echo "  git config --global credential.helper store"
echo "  git push"
echo "  (输入用户名和PAT作为密码)"
echo ""

echo "方法2: 使用SSH（推荐）"
echo "----------------------------------------"
echo "1. 生成SSH密钥："
echo "   ssh-keygen -t ed25519 -C \"your_email@example.com\""
echo ""
echo "2. 添加SSH密钥到GitHub："
echo "   cat ~/.ssh/id_ed25519.pub"
echo "   (复制输出到 GitHub Settings > SSH and GPG keys)"
echo ""
echo "3. 测试连接："
echo "   ssh -T git@github.com"
echo ""

read -p "选择方法 (1=HTTPS, 2=SSH, 3=直接尝试推送): " -n 1 -r
echo ""

case $REPLY in
    1)
        echo "使用HTTPS推送..."
        git remote set-url origin https://github.com/RyanLi888/MEDAL.git
        git push
        ;;
    2)
        echo "使用SSH推送..."
        git remote set-url origin git@github.com:RyanLi888/MEDAL.git
        
        # 检查SSH密钥
        if [ ! -f ~/.ssh/id_ed25519 ] && [ ! -f ~/.ssh/id_rsa ]; then
            echo "⚠️  未找到SSH密钥，是否生成？(y/n): "
            read -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                EMAIL=$(git config user.email)
                ssh-keygen -t ed25519 -C "$EMAIL" -f ~/.ssh/id_ed25519 -N ""
                echo ""
                echo "✓ SSH密钥已生成，请添加到GitHub："
                echo "  cat ~/.ssh/id_ed25519.pub"
                echo ""
                echo "然后运行: ssh -T git@github.com 测试连接"
                exit 0
            fi
        fi
        
        git push
        ;;
    3)
        echo "直接尝试推送..."
        git push
        ;;
    *)
        echo "❌ 已取消"
        exit 1
        ;;
esac

echo ""
echo "✓ 推送完成！"
