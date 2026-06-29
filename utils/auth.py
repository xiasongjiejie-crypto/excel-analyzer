"""轻量级密码认证。

基于 Streamlit secrets 机制：密码存储在 `.streamlit/secrets.toml`（本地）
或 Streamlit Cloud 后台（部署），不会出现在代码或 Git 仓库中。

用法（在 app.py 主函数开头）：
    from utils import auth
    if not auth.check_password():
        st.stop()
"""
from __future__ import annotations

import hmac

import streamlit as st


def check_password() -> bool:
    """校验用户输入的访问密码。

    行为：
    - 若未配置密码（secrets 中无 "password"），视为无保护模式直接放行，
      并给出提示（方便本地开发）。
    - 已配置时，要求用户输入正确密码后才返回 True。

    Returns:
        True 表示验证通过（或无需验证），False 表示尚未通过。
    """
    # 未配置密码 -> 本地开发无保护模式
    if "password" not in st.secrets:
        st.warning(
            "⚠️ 当前未设置访问密码，应用处于**无保护模式**。"
            "公网部署请在 Streamlit Cloud 后台的 Secrets 中配置 `password`。"
        )
        return True

    # 已验证通过
    if st.session_state.get("password_correct", False):
        return True

    def _on_submit() -> None:
        """回调：比对输入密码，成功后清除明文。"""
        if hmac.compare_digest(
            str(st.session_state.get("password_input", "")),
            str(st.secrets["password"]),
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password_input"]  # 不在内存中保留明文
        else:
            st.session_state["password_correct"] = False

    # 渲染登录界面
    st.title("🔒 访问验证")
    st.text_input(
        "请输入访问密码", type="password",
        on_change=_on_submit, key="password_input",
    )
    if st.session_state.get("password_correct") is False:
        st.error("密码错误，请重试。")
    return False
