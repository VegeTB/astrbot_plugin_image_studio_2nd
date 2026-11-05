"""多服务商文生图与图生图插件。

功能：
- 聊天指令 /文生图、/t2i 与 /图生图、/i2i。
- 向 AI 智能体注册 text_to_image、image_to_image、svg_render、create_animation 四个工具。
- 可视化设置页：选择提供商、填写 Base URL 与 API Key、检测可用模型、
  文生图/图生图测试，测试通过后一键保存配置。
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path

import httpx
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.message_components import File, Image, Reply, Video
from astrbot.api.star import Context, Star, register
from astrbot.api.web import error_response, json_response, request
from astrbot.core.star.filter.command import GreedyStr
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

from .client import ImageProviderError, generate_image, list_models
from .motion_tool import MotionError, render_animation
from .profiles import ProfileStore
from .providers import PROVIDERS, PROVIDER_IDS, resolve_provider
from .svg_tool import SVGError, render_svg_to_png

PLUGIN_NAME = "astrbot_plugin_image_studio"
TOOL_NAMES = ["text_to_image", "image_to_image", "svg_render", "create_animation"]

# 插件级全局配置（多服务商档案保存在 profiles.json）。
GLOBAL_KEYS = [
    "allow_commands",
    "register_llm_tools",
    "allowed_users",
    "svg_width",
    "svg_height",
    "svg_background",
    "svg_max_dimension",
    "motion_default_format",
    "motion_width",
    "motion_height",
    "motion_fps",
    "motion_max_duration",
    "motion_max_frames",
    "motion_max_dimension",
    "motion_max_file_mb",
    "motion_video_as_file",
    "motion_max_concurrency",
]

_MIME_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}

ANIMATION_HELP = """用本地 SVG 动画解释概念。支持 flow、timeline、cycle、comparison、text、chart、slideshow。
自然语言直接告诉 AI 即可；手动指令请提供 JSON，例如：
{"pattern":"flow","title":"TCP 三次握手","actors":[{"id":"client","label":"客户端"},{"id":"server","label":"服务器"}],"steps":[{"from":"client","to":"server","label":"SYN"},{"from":"server","to":"client","label":"SYN + ACK"},{"from":"client","to":"server","label":"ACK"}]}
可选 format: gif、webp、mp4。slideshow 需要附带图片。"""


def unwrap_motion_spec(spec):
    """兼容部分模型把动画脚本包在 {"spec": {...}} 中的写法。"""
    if isinstance(spec, dict) and not spec.get("pattern") and isinstance(spec.get("spec"), dict):
        return spec["spec"]
    return spec


def resolve_config(config: dict) -> dict:
    """把插件配置解析成可直接调用的客户端配置。

    当用户选择内置提供商但 Base URL 仍是其它提供商的默认值时，自动纠正为
    所选提供商的地址；apikey.fan 需要流式返回，因此强制开启 stream。
    """
    cfg = dict(config)
    provider_id = str(cfg.get("provider") or "custom")
    preset = resolve_provider(provider_id)

    configured_url = (cfg.get("base_url") or "").strip().rstrip("/")
    preset_url = (preset.get("base_url") or "").strip().rstrip("/")
    if provider_id != "custom":
        other_urls = {
            (p.get("base_url") or "").strip().rstrip("/")
            for p in PROVIDERS
            if (p.get("base_url") or "").strip()
        }
        if not configured_url or (
            configured_url in other_urls and configured_url != preset_url
        ):
            cfg["base_url"] = preset_url

    if not (cfg.get("model") or "").strip():
        cfg["model"] = preset.get("default_model") or ""

    # 按提供商能力收敛参数。
    if not bool(preset.get("supports_stream", True)):
        cfg["stream"] = False
    elif provider_id == "apikey":
        cfg["stream"] = True

    max_n = int(preset.get("max_n") or 4)
    try:
        cfg["n"] = max(1, min(int(cfg.get("n") or 1), max_n))
    except (TypeError, ValueError):
        cfg["n"] = 1

    cfg["supports_response_format"] = bool(
        preset.get("supports_response_format", True)
    )
    cfg["supports_output_format"] = bool(
        preset.get("supports_output_format", False)
    )
    cfg["output_format"] = str(cfg.get("output_format") or "png").strip().lower()
    if cfg.get("output_compression") is None:
        cfg["output_compression"] = 90

    return cfg


@register(
    PLUGIN_NAME,
    "xky201225",
    "多服务商文生图、图生图、SVG矢量渲染与概念动画生成插件",
    "1.1.0",
)
class ImageStudioPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None):
        try:
            super().__init__(context, config)
        except TypeError:
            super().__init__(context)
        self.config = config if config is not None else {}

        self.profile_store = ProfileStore(self._data_dir())
        self.profile_store.ensure_from_legacy(dict(self.config))
        self._motion_semaphore: asyncio.Semaphore | None = None

        self._register_web_apis()
        self._ensure_llm_tools_registered()
        logger.info(f"{PLUGIN_NAME} 已加载。")

    @filter.on_astrbot_loaded()
    async def _on_astrbot_loaded(self):
        """当 AstrBot 所有模块和人设加载完毕后，再次确保工具处于激活并可见状态。"""
        self._ensure_llm_tools_registered()

    def _ensure_llm_tools_registered(self) -> None:
        """强制对齐工具模块路径、激活工具并同步至人设白名单，解决工具不显示问题。"""
        if not self.config.get("register_llm_tools", True):
            return

        main_module_path = self.__class__.__module__
        plugin_pkg = main_module_path.rsplit(".", 1)[0]

        # 1. 修正 star_handlers_registry 中的模块路径
        try:
            from astrbot.core.star.star_handler import star_handlers_registry
            for handler_meta in star_handlers_registry:
                if handler_meta.handler_module_path.startswith(plugin_pkg):
                    handler_meta.handler_module_path = main_module_path
        except Exception as exc:
            logger.debug(f"[{PLUGIN_NAME}] 修正 handler_registry 路径跳过: {exc}")

        # 2. 激活 func_tool_manager 中的工具并绑定实例方法
        try:
            tool_mgr = self.context.get_llm_tool_manager()
            if tool_mgr and hasattr(tool_mgr, "func_list"):
                active_names = []
                for tool in tool_mgr.func_list:
                    t_name = getattr(tool, "name", "")
                    if t_name in TOOL_NAMES:
                        tool.handler_module_path = main_module_path
                        tool.active = True
                        if hasattr(self, t_name):
                            tool.handler = getattr(self, t_name)
                        active_names.append(t_name)
                logger.info(f"[{PLUGIN_NAME}] 已确认激活 LLM 工具: {active_names}")
        except Exception as exc:
            logger.warning(f"[{PLUGIN_NAME}] 激活 LLM 工具时出现异常: {exc}")

        # 3. 如果用户的人设配置了局部工具白名单，自动把本插件的 4 个工具加入白名单
        try:
            pm = getattr(self.context, "provider_manager", None)
            personas = getattr(pm, "personas", None) or []
            for p in personas:
                if isinstance(p, dict):
                    tools_list = p.get("tools")
                    if isinstance(tools_list, list):
                        for t_name in TOOL_NAMES:
                            if t_name not in tools_list:
                                tools_list.append(t_name)
                else:
                    tools_list = getattr(p, "tools", None)
                    if isinstance(tools_list, list):
                        for t_name in TOOL_NAMES:
                            if t_name not in tools_list:
                                tools_list.append(t_name)
        except Exception as exc:
            logger.debug(f"[{PLUGIN_NAME}] 同步人设工具白名单跳过: {exc}")

    # ------------------------------------------------------------------ #
    # 基础工具方法
    # ------------------------------------------------------------------ #
    def _data_dir(self) -> Path:
        path = Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _runtime_cfg(self) -> dict:
        try:
            profile = self.profile_store.get_active()
        except Exception:  # noqa: BLE001
            profile = dict(self.config)
        return resolve_config(profile)

    def _check_allowed(self, event: AstrMessageEvent) -> bool:
        allowed = self.config.get("allowed_users") or []
        if not allowed:
            return True
        return event.get_sender_id() in allowed

    async def _gather_image_paths(
        self, event: AstrMessageEvent, limit: int = 3
    ) -> list[str]:
        """递归收集消息链（含引用消息）中的图片，并转为本地文件路径。"""
        found: list[Image] = []

        def walk(items):
            for comp in items or []:
                if isinstance(comp, Image):
                    found.append(comp)
                elif isinstance(comp, Reply) and getattr(comp, "chain", None):
                    walk(comp.chain)

        walk(event.get_messages())
        paths: list[str] = []
        for img in found[:limit]:
            try:
                paths.append(await img.convert_to_file_path())
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"转换图片失败：{exc}")
        return paths

    async def _materialize_images(self, images: list[dict]) -> list[Path]:
        """把 b64/url 图片统一落盘，返回本地文件路径列表。"""
        out_dir = self._data_dir() / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        paths: list[Path] = []
        for idx, img in enumerate(images):
            try:
                if img.get("b64"):
                    mime = img.get("mime") or "image/png"
                    ext = _MIME_EXT.get(mime, "png")
                    data = base64.b64decode(img["b64"])
                    target = out_dir / f"img_{stamp}_{idx}.{ext}"
                    target.write_bytes(data)
                    paths.append(target)
                elif img.get("url"):
                    async with httpx.AsyncClient(
                        timeout=120, follow_redirects=True
                    ) as client:
                        resp = await client.get(img["url"])
                        resp.raise_for_status()
                    ext = "png"
                    target = out_dir / f"img_{stamp}_{idx}.{ext}"
                    target.write_bytes(resp.content)
                    paths.append(target)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"保存第 {idx + 1} 张图片失败：{exc}")
        return paths

    async def _generate_and_save(
        self, prompt: str, image_paths: list[str] | None = None
    ) -> list[Path]:
        cfg = self._runtime_cfg()
        if not (cfg.get("base_url") or "").strip():
            raise ImageProviderError("请先在插件配置中填写 Base URL")
        if not (cfg.get("api_key") or "").strip():
            raise ImageProviderError("请先在插件配置中填写 API Key")
        images = await generate_image(cfg, prompt, image_paths)
        paths = await self._materialize_images(images)
        if not paths:
            raise ImageProviderError("图片生成成功但保存失败")
        return paths

    async def _extract_svg_input(self, event: AstrMessageEvent, text: str) -> str:
        """优先使用指令后的 SVG 文本，否则读取消息中的 .svg 文件。"""
        source = (text or "").strip()
        if source:
            return source

        found: list = []

        def walk(items):
            for comp in items or []:
                type_text = str(getattr(comp, "type", "")).lower()
                class_name = comp.__class__.__name__.lower()
                if "file" in type_text or class_name == "file":
                    found.append(comp)
                elif getattr(comp, "chain", None):
                    walk(comp.chain)

        walk(event.get_messages())
        for comp in found:
            name = str(getattr(comp, "name", "") or getattr(comp, "file", "") or "")
            if not name.lower().endswith(".svg"):
                continue
            converter = getattr(comp, "convert_to_file_path", None)
            if not converter:
                continue
            try:
                path = await converter()
                return Path(path).read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"读取 SVG 文件失败：{exc}")
        return ""

    async def _render_svg_and_save(self, svg: str) -> Path:
        width = int(self.config.get("svg_width") or 0)
        height = int(self.config.get("svg_height") or 0)
        background = str(self.config.get("svg_background") or "#ffffff")
        max_dim = int(self.config.get("svg_max_dimension") or 4096)
        png, _out_w, _out_h = await asyncio.to_thread(
            render_svg_to_png, svg, width, height, background, max_dim
        )
        out_dir = self._data_dir() / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"svg_{int(time.time() * 1000)}.png"
        target.write_bytes(png)
        return target

    def _motion_defaults(self) -> dict:
        return {
            "format": str(self.config.get("motion_default_format") or "gif"),
            "width": int(self.config.get("motion_width") or 640),
            "height": int(self.config.get("motion_height") or 360),
            "fps": int(self.config.get("motion_fps") or 12),
            "max_duration": float(self.config.get("motion_max_duration") or 15),
            "max_frames": int(self.config.get("motion_max_frames") or 360),
            "max_dimension": int(self.config.get("motion_max_dimension") or 1280),
            "max_file_mb": int(self.config.get("motion_max_file_mb") or 20),
        }

    def _cleanup_motion_outputs(self, max_age_hours: int = 24) -> None:
        out_dir = self._data_dir() / "outputs"
        if not out_dir.exists():
            return
        cutoff = time.time() - max_age_hours * 3600
        for path in out_dir.glob("motion_*"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue

    async def _render_motion(
        self, spec: dict, event: AstrMessageEvent | None = None
    ) -> dict:
        if self._motion_semaphore is None:
            self._motion_semaphore = asyncio.Semaphore(
                max(1, min(4, int(self.config.get("motion_max_concurrency") or 1)))
            )
        images = await self._gather_image_paths(event, limit=12) if event else []
        out_dir = self._data_dir() / "outputs"
        self._cleanup_motion_outputs()
        async with self._motion_semaphore:
            return await asyncio.to_thread(
                render_animation,
                spec,
                out_dir,
                images,
                self._motion_defaults(),
            )

    def _send_motion(self, event: AstrMessageEvent, result: dict):
        path = Path(result["path"])
        fmt = result["format"]
        force_file = bool(result.get("send_as_file")) or (
            fmt == "mp4" and bool(self.config.get("motion_video_as_file"))
        )
        if force_file:
            yield event.chain_result([File(file=str(path), name=path.name)])
        elif fmt == "mp4":
            yield event.chain_result([Video.fromFileSystem(str(path))])
        else:
            yield event.image_result(str(path))
        if result.get("caption", True):
            yield event.plain_result(
                f"已生成 {result['duration']:.1f} 秒、{result['width']}×{result['height']} 的 {fmt.upper()} 动画。"
            )

    # ------------------------------------------------------------------ #
    # 聊天指令
    # ------------------------------------------------------------------ #
    @filter.command("文生图", alias={"t2i", "draw"})
    async def cmd_text_to_image(self, event: AstrMessageEvent, prompt: GreedyStr):
        """文生图指令：/文生图 一只在太空中的猫"""
        if not self.config.get("allow_commands", True):
            yield event.plain_result("文生图指令未启用，请先在插件配置中开启。")
            return
        if not self._check_allowed(event):
            yield event.plain_result("你没有使用该指令的权限。")
            return
        prompt = (prompt or "").strip()
        if not prompt:
            yield event.plain_result("用法：/文生图 <描述>，例如 /文生图 一只在太空中的猫")
            return
        yield event.plain_result("正在生成图片，请稍候…")
        try:
            paths = await self._generate_and_save(prompt)
            for path in paths:
                yield event.image_result(str(path))
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"文生图失败：{exc}")

    @filter.command("图生图", alias={"i2i", "edit"})
    async def cmd_image_to_image(self, event: AstrMessageEvent, prompt: GreedyStr):
        """图生图指令：发送一张图片并附上 /图生图 描述"""
        if not self.config.get("allow_commands", True):
            yield event.plain_result("图生图指令未启用，请先在插件配置中开启。")
            return
        if not self._check_allowed(event):
            yield event.plain_result("你没有使用该指令的权限。")
            return
        prompt = (prompt or "").strip()
        if not prompt:
            yield event.plain_result("用法：发送一张图片，并附上文字 /图生图 <修改描述>")
            return
        yield event.plain_result("正在生成图片，请稍候…")
        try:
            refs = await self._gather_image_paths(event)
            if not refs:
                yield event.plain_result("没有找到参考图片，请发送一张图片并附上 /图生图 描述。")
                return
            paths = await self._generate_and_save(prompt, refs)
            for path in paths:
                yield event.image_result(str(path))
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"图生图失败：{exc}")

    @filter.command("svg绘图", alias={"svgdraw", "矢量绘图"})
    async def cmd_svg_draw(self, event: AstrMessageEvent, svg: GreedyStr):
        """SVG 绘图指令：/svg绘图 <完整 SVG 代码>，或附带 .svg 文件。"""
        if not self.config.get("allow_commands", True):
            yield event.plain_result("SVG 绘图指令未启用，请先在插件配置中开启。")
            return
        if not self._check_allowed(event):
            yield event.plain_result("你没有使用该指令的权限。")
            return
        source = await self._extract_svg_input(event, svg)
        if not source:
            yield event.plain_result(
                "请提供完整 SVG 代码，例如 /svg绘图 <svg xmlns=...>，或附带一个 .svg 文件。"
            )
            return
        yield event.plain_result("正在渲染 SVG…")
        try:
            path = await self._render_svg_and_save(source)
            yield event.image_result(str(path))
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"SVG 绘图失败：{exc}")

    @filter.command("动画", alias={"animation", "motion"})
    async def cmd_animation(self, event: AstrMessageEvent, spec_text: GreedyStr):
        """动画说明指令：/动画 <JSON 动画脚本>。自然语言可由 AI 调用 create_animation。"""
        if not self.config.get("allow_commands", True):
            yield event.plain_result("动画指令未启用，请先在插件配置中开启。")
            return
        if not self._check_allowed(event):
            yield event.plain_result("你没有使用该指令的权限。")
            return
        text = (spec_text or "").strip()
        if not text or text.lower() in ("帮助", "help", "?"):
            yield event.plain_result(ANIMATION_HELP)
            return
        try:
            spec = json.loads(text)
        except json.JSONDecodeError as exc:
            yield event.plain_result(
                f"动画脚本不是合法 JSON：{exc}\n发送 /动画 帮助 查看示例。"
            )
            return
        if not isinstance(spec, dict):
            yield event.plain_result("动画脚本必须是 JSON 对象。")
            return
        spec = unwrap_motion_spec(spec)
        yield event.plain_result("正在生成动画，请稍候…")
        try:
            result = await self._render_motion(spec, event)
            for item in self._send_motion(event, result):
                yield item
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"动画生成失败：{exc}")

    # ------------------------------------------------------------------ #
    # LLM 工具（供 AstrBot 智能体调用，兼容 Anthropic / OpenAI 协议）
    # ------------------------------------------------------------------ #
    @filter.llm_tool(name="text_to_image")
    async def text_to_image(self, event: AstrMessageEvent, prompt: str):
        """根据文字描述生成图片并直接发送给用户。

        Args:
            prompt(string): 图片的文字描述，例如“一只穿着宇航服的猫”。
        """
        if not self.config.get("register_llm_tools", True):
            yield event.plain_result("文生图工具未启用，请先在插件配置中开启。")
            return
        try:
            paths = await self._generate_and_save(prompt.strip())
            for path in paths:
                yield event.image_result(str(path))
            if not paths:
                yield event.plain_result("没有生成任何图片。")
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"文生图失败：{exc}")

    @filter.llm_tool(name="image_to_image")
    async def image_to_image(self, event: AstrMessageEvent, prompt: str):
        """根据用户当前消息附带的参考图片进行图生图（改图），并直接发送结果。

        Args:
            prompt(string): 对参考图片的修改描述，例如“把背景换成星空”。
        """
        if not self.config.get("register_llm_tools", True):
            yield event.plain_result("图生图工具未启用，请先在插件配置中开启。")
            return
        try:
            refs = await self._gather_image_paths(event)
            if not refs:
                yield event.plain_result("没有找到参考图片，请先发送一张图片。")
                return
            paths = await self._generate_and_save(prompt.strip(), refs)
            for path in paths:
                yield event.image_result(str(path))
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"图生图失败：{exc}")

    @filter.llm_tool(name="svg_render")
    async def svg_render(self, event: AstrMessageEvent, svg: str):
        """把完整 SVG 代码渲染成 PNG 图片并直接发送，用于流程图、图表、示意图、图标和海报等本地矢量绘图。

        Args:
            svg(string): 完整的 SVG 文档，必须包含根节点 <svg>、xmlns、width、height 和 viewBox。
        """
        if not self.config.get("register_llm_tools", True):
            yield event.plain_result("SVG 绘图工具未启用，请先在插件配置中开启。")
            return
        try:
            path = await self._render_svg_and_save(svg)
            yield event.image_result(str(path))
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"SVG 绘图失败：{exc}")

    @filter.llm_tool(name="create_animation")
    async def create_animation(self, event: AstrMessageEvent, spec: str):
        """用本地 SVG 动画解释一个概念，并把 GIF/WebP/MP4 直接发送给用户。

        不要调用文生视频模型。你负责把用户描述转成动画脚本 JSON 字符串，工具负责逐帧渲染和编码。
        先判断用户想解释的是流程、时间顺序、循环、对比、文字强调、数据还是图片轮播，再选择 pattern。
        每个动画保持 3 到 6 个关键节点，节点标签尽量简短（2 到 10 个字），不要堆叠大量元素。
        默认 format=gif；预计超过 8 秒或内容较复杂时用 mp4。slideshow 只在用户提供了图片时使用。
        pattern 可选：
        - flow: 流程/因果，需要 actors[{id,label}] 和 steps[{from,to,label,description}]
        - timeline: 时间线，需要 events[{time,label,description}]
        - cycle: 循环，需要 nodes["蒸发","凝结",...] 和可选 center
        - comparison: 对比，需要 left{title,items} 与 right{title,items}
        - text: 文字动效，需要 lines["第一行",...] 和可选 style(fade/slide/typewriter/neon)
        - chart: 图表，需要 chart_type(bar/line)、labels、values 或 series[{name,values}]
        - slideshow: 图片轮播，使用当前消息附带的图片，可选 transition(fade/slide/zoom)
        通用字段：title、subtitle、format(gif/webp/mp4)、width、height、fps、duration、theme(dark/light/ocean/sunset)、accent、background、loop、send_as_file(平台不支持视频时以文件发送)、caption(是否附带生成摘要)。

        Args:
            spec(string): 完整的动画 JSON 脚本字符串，例如 {"pattern":"flow","title":"TCP 三次握手","actors":[{"id":"client","label":"客户端"},{"id":"server","label":"服务器"}],"steps":[{"from":"client","to":"server","label":"SYN"}]}。
        """
        if not self.config.get("register_llm_tools", True):
            yield event.plain_result("动画工具未启用，请先在插件配置中开启。")
            return
        if isinstance(spec, str):
            try:
                spec = json.loads(spec)
            except json.JSONDecodeError as exc:
                yield event.plain_result(f"动画脚本不是合法 JSON：{exc}")
                return
        if not isinstance(spec, dict):
            yield event.plain_result("动画脚本必须是 JSON 对象。")
            return
        spec = unwrap_motion_spec(spec)
        try:
            result = await self._render_motion(spec, event)
            for item in self._send_motion(event, result):
                yield item
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"动画生成失败：{exc}")

    # ------------------------------------------------------------------ #
    # 可视化设置页后端 API
    # ------------------------------------------------------------------ #
    def _register_web_apis(self) -> None:
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/providers", self.api_providers, ["GET"], "获取提供商预设"
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/config", self.api_config, ["GET"], "读取当前配置"
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/config/save", self.api_save_config, ["POST"], "保存配置"
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/models", self.api_detect_models, ["POST"], "检测可用模型"
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/test", self.api_test, ["POST"], "测试文生图/图生图"
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/profiles", self.api_profiles, ["GET"], "获取配置档案"
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/profiles/save", self.api_profile_save, ["POST"], "保存配置档案"
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/profiles/delete", self.api_profile_delete, ["POST"], "删除配置档案"
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/profiles/activate", self.api_profile_activate, ["POST"], "切换配置档案"
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/profiles/test", self.api_profile_test, ["POST"], "测试配置档案"
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/svg/render", self.api_svg_render, ["POST"], "渲染 SVG 为 PNG"
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/animation/render", self.api_animation_render, ["POST"], "渲染动画"
        )

    async def api_providers(self):
        return json_response({"providers": PROVIDERS, "ids": PROVIDER_IDS})

    async def api_config(self):
        return json_response(
            {"config": {key: self.config.get(key) for key in GLOBAL_KEYS}}
        )

    async def api_save_config(self):
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            return error_response("请求体必须是 JSON 对象", status_code=400)
        for key in GLOBAL_KEYS:
            if key in payload:
                self.config[key] = payload[key]
        self.config["allow_commands"] = bool(self.config.get("allow_commands", True))
        self.config["register_llm_tools"] = bool(
            self.config.get("register_llm_tools", True)
        )
        if not isinstance(self.config.get("allowed_users"), list):
            self.config["allowed_users"] = []
        for key in ("svg_width", "svg_height"):
            try:
                self.config[key] = max(0, min(8192, int(self.config.get(key) or 0)))
            except (TypeError, ValueError):
                self.config[key] = 0
        try:
            self.config["svg_max_dimension"] = max(
                64, min(8192, int(self.config.get("svg_max_dimension") or 4096))
            )
        except (TypeError, ValueError):
            self.config["svg_max_dimension"] = 4096
        self.config["svg_background"] = (
            str(self.config.get("svg_background") or "#ffffff").strip() or "#ffffff"
        )
        if self.config.get("motion_default_format") not in ("gif", "webp", "mp4"):
            self.config["motion_default_format"] = "gif"
        for key, default, low, high in (
            ("motion_width", 640, 240, 1280),
            ("motion_height", 360, 240, 1280),
            ("motion_fps", 12, 5, 24),
            ("motion_max_duration", 15, 2, 60),
            ("motion_max_frames", 360, 10, 600),
            ("motion_max_dimension", 1280, 240, 4096),
            ("motion_max_file_mb", 20, 1, 100),
            ("motion_max_concurrency", 1, 1, 4),
        ):
            try:
                self.config[key] = max(
                    low, min(high, int(self.config.get(key) or default))
                )
            except (TypeError, ValueError):
                self.config[key] = default
        self.config["motion_video_as_file"] = bool(
            self.config.get("motion_video_as_file")
        )
        await self.config.save_config_async()
        self._ensure_llm_tools_registered()
        return json_response(
            {
                "ok": True,
                "config": {key: self.config.get(key) for key in GLOBAL_KEYS},
            }
        )

    async def api_detect_models(self):
        payload = await request.json(default={})
        base_url = (payload.get("base_url") or "").strip()
        api_key = (payload.get("api_key") or "").strip()
        if not base_url:
            return error_response("请先填写 Base URL", status_code=400)
        cfg = {
            "base_url": base_url,
            "api_key": api_key,
            "timeout": payload.get("timeout") or 60,
        }
        try:
            models = await list_models(cfg)
            return json_response({"ok": True, "models": models})
        except ImageProviderError as exc:
            return error_response(str(exc), status_code=400)

    async def api_test(self):
        payload = await request.json(default={})
        return await self._run_test(payload)

    async def api_profile_test(self):
        payload = await request.json(default={})
        return await self._run_test(payload)

    async def _run_test(self, payload: dict):
        if not isinstance(payload, dict):
            return error_response("请求体必须是 JSON 对象", status_code=400)
        mode = payload.get("mode") or "text"
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return error_response("请先填写测试 prompt", status_code=400)

        cfg = {
            "provider": payload.get("provider"),
            "base_url": payload.get("base_url"),
            "api_key": payload.get("api_key"),
            "model": payload.get("model"),
            "size": payload.get("size") or "1024x1024",
            "quality": payload.get("quality") or "auto",
            "n": payload.get("n") or 1,
            "stream": bool(payload.get("stream")),
            "response_format": payload.get("response_format") or "b64_json",
            "output_format": payload.get("output_format") or "png",
            "output_compression": payload.get("output_compression"),
            "t2i_path": payload.get("t2i_path") or "/images/generations",
            "i2i_path": payload.get("i2i_path") or "/images/edits",
            "timeout": payload.get("timeout") or 300,
        }
        cfg = resolve_config(cfg)

        image_paths: list[str] = []
        if mode == "image":
            image_paths = await self._resolve_test_image(payload)
            if not image_paths:
                return error_response("请先上传参考图片", status_code=400)

        try:
            images = await generate_image(cfg, prompt, image_paths)
            return json_response({"ok": True, "images": images})
        except ImageProviderError as exc:
            return error_response(str(exc), status_code=400)

    async def api_profiles(self):
        data = self.profile_store.load()
        return json_response(
            {"profiles": data["profiles"], "active_id": data["active_id"]}
        )

    async def api_profile_save(self):
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            return error_response("请求体必须是 JSON 对象", status_code=400)
        set_active = bool(payload.pop("set_active", False))
        if not str(payload.get("base_url") or "").strip():
            return error_response("请先填写 Base URL", status_code=400)
        profile, data = self.profile_store.upsert(payload, set_active=set_active)
        return json_response(
            {
                "ok": True,
                "saved_id": profile["id"],
                "profiles": data["profiles"],
                "active_id": data["active_id"],
            }
        )

    async def api_profile_delete(self):
        payload = await request.json(default={})
        profile_id = str(payload.get("id") or "").strip()
        if not profile_id:
            return error_response("缺少配置 id", status_code=400)
        data = self.profile_store.delete(profile_id)
        return json_response(
            {
                "ok": True,
                "profiles": data["profiles"],
                "active_id": data["active_id"],
            }
        )

    async def api_profile_activate(self):
        payload = await request.json(default={})
        profile_id = str(payload.get("id") or "").strip()
        if not profile_id:
            return error_response("缺少配置 id", status_code=400)
        data = self.profile_store.activate(profile_id)
        return json_response(
            {
                "ok": True,
                "profiles": data["profiles"],
                "active_id": data["active_id"],
            }
        )

    async def api_svg_render(self):
        payload = await request.json(default={})
        svg = str(payload.get("svg") or "").strip()
        if not svg:
            return error_response("请提供 SVG 代码", status_code=400)
        try:
            width = int(payload.get("width") or self.config.get("svg_width") or 0)
            height = int(payload.get("height") or self.config.get("svg_height") or 0)
            max_dim = int(
                payload.get("max_dimension")
                or self.config.get("svg_max_dimension")
                or 4096
            )
        except (TypeError, ValueError):
            return error_response("宽度、高度或最大边长必须是整数", status_code=400)
        background = str(
            payload.get("background")
            or self.config.get("svg_background")
            or "#ffffff"
        )
        try:
            png, out_w, out_h = await asyncio.to_thread(
                render_svg_to_png, svg, width, height, background, max_dim
            )
        except SVGError as exc:
            return error_response(str(exc), status_code=400)
        return json_response(
            {
                "ok": True,
                "mime": "image/png",
                "b64": base64.b64encode(png).decode(),
                "width": out_w,
                "height": out_h,
            }
        )

    async def api_animation_render(self):
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            return error_response("动画脚本必须是 JSON 对象", status_code=400)
        payload = unwrap_motion_spec(payload)
        try:
            result = await self._render_motion(payload)
        except MotionError as exc:
            return error_response(str(exc), status_code=400)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Animation render failed")
            return error_response(f"动画渲染失败：{exc}", status_code=500)
        path = Path(result["path"])
        try:
            data = path.read_bytes()
        except OSError as exc:
            return error_response(f"读取动画文件失败：{exc}", status_code=500)
        finally:
            path.unlink(missing_ok=True)
        return json_response(
            {
                "ok": True,
                "b64": base64.b64encode(data).decode(),
                "mime": result["mime"],
                "filename": Path(result["path"]).name,
                "width": result["width"],
                "height": result["height"],
                "frames": result["frames"],
                "duration": result["duration"],
                "size_bytes": result["size_bytes"],
            }
        )

    async def _resolve_test_image(self, payload: dict) -> list[str]:
        tmp_dir = self._data_dir() / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        b64 = payload.get("image_b64") or ""
        if b64:
            if "," in b64 and b64.lstrip().lower().startswith("data:"):
                b64 = b64.split(",", 1)[1]
            try:
                data = base64.b64decode(b64)
            except Exception:
                return []
            target = tmp_dir / f"ref_{int(time.time() * 1000)}.png"
            target.write_bytes(data)
            return [str(target)]

        url = (payload.get("image_url") or "").strip()
        if url:
            try:
                async with httpx.AsyncClient(
                    timeout=60, follow_redirects=True
                ) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                target = tmp_dir / f"ref_{int(time.time() * 1000)}.png"
                target.write_bytes(resp.content)
                return [str(target)]
            except Exception:
                return []
        return []

    async def terminate(self):
        """插件卸载/停用时调用。"""
        logger.info(f"{PLUGIN_NAME} 已卸载。")
