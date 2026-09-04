# 用 uv 管理 product_category_match 的环境

## Context

当前项目没有任何环境定义文件，只有一份 `requirements.txt`：它是 PowerShell 下 `pip freeze >` 生成的产物，因此是 **UTF-16 编码**（很多工具读起来是乱码），并且把 79 个包全部平铺——既有 `certifi` / `idna` / `jinja2` 这类传递依赖，也有 `ipykernel` / `debugpy` / `jedi` 这类只服务于 [play.ipynb](play.ipynb) 的开发工具，还混进了 `dotenv==0.9.9` 这个和代码里实际用的 `python-dotenv` 不是一回事的错误包。README 让使用者"随便建个 conda/venv 然后 pip install -r"，没有锁文件，换台机器结果不可复现。

目标：改用 uv 管理——`pyproject.toml` 声明**直接依赖**，`uv.lock` 精确锁定完整依赖树，`.python-version` 固定解释器版本，一条 `uv sync` 就能得到可复现的环境。torch 走默认 PyPI（Windows 上即 CPU-only wheel），不配置 CUDA index。

本机已有 uv 0.12.0，全局 Python 3.12.9，项目当前**没有** `.venv`（`tools/__pycache__/*.cpython-312.pyc` 也印证了原环境是 3.12）。

## 依赖梳理

代码里真正 import 的直接依赖（版本下限取自现有 freeze，保证不低于已验证过的版本）：

| 包 | 来源 |
|---|---|
| `pandas>=2.3.2` | [main.py:1](main.py#L1) |
| `numpy>=2.3.2` | [main.py:4](main.py#L4) |
| `joblib>=1.5.1` | [main.py:5](main.py#L5) |
| `tqdm>=4.67.1` | [main.py:3](main.py#L3) |
| `pyyaml>=6.0.2` | [main.py:12](main.py#L12) |
| `openpyxl>=3.1.5` | `pd.read_excel` / `to_excel`（[main.py:32-33](main.py#L32-L33)、[main.py:276](main.py#L276)），pandas 不会自动装 |
| `sentence-transformers>=5.1.0` | [tools/match_func.py:2](tools/match_func.py#L2)，间接带来 torch / transformers / scikit-learn / scipy / huggingface-hub / pillow |
| `openai>=1.101.0` | [tools/trans.py:3](tools/trans.py#L3) |
| `python-dotenv>=1.1.1` | [tools/trans.py:1](tools/trans.py#L1)（丢弃 freeze 里的 `dotenv==0.9.9`） |

dev 组：`ipykernel>=6.30.1`（供 [play.ipynb](play.ipynb) 使用）。其余 68 个包都是传递依赖，交给 `uv.lock` 锁定，不写进 `pyproject.toml`。

## 实施步骤

### 1. 新建 `pyproject.toml`（项目根目录）

```toml
[project]
name = "product-category-match"
version = "0.1.0"
description = "Match or update product categories for online retail data"
readme = "README.md"
requires-python = ">=3.12"
license = { file = "LICENSE" }
dependencies = [
    "joblib>=1.5.1",
    "numpy>=2.3.2",
    "openai>=1.101.0",
    "openpyxl>=3.1.5",
    "pandas>=2.3.2",
    "python-dotenv>=1.1.1",
    "pyyaml>=6.0.2",
    "sentence-transformers>=5.1.0",
    "tqdm>=4.67.1",
]

[dependency-groups]
dev = ["ipykernel>=6.30.1"]

[tool.uv]
package = false
```

`package = false` 很关键：项目是脚本式布局（根目录直接放 `main.py`、`tools/`），没有可安装的包结构，加上它 uv 就只管理依赖、不尝试 build 本项目。

### 2. 新建 `.python-version`

内容一行：`3.12`。让 `uv sync` 在任何机器上都取 3.12 解释器（本机已有 3.12.9；若缺失 uv 会自行下载）。

### 3. 生成锁文件与虚拟环境

```powershell
uv sync
```

会在项目根创建 `.venv/` 并写出 `uv.lock`。torch 从默认 PyPI 拉取 Windows CPU wheel。

> 注意：项目位于 OneDrive 同步目录下，`.venv/` 有上万个小文件。装完建议在 OneDrive 设置里把 `.venv` 标记为"始终保留在此设备上"或排除同步，否则同步会很慢——这不影响功能，只是体验建议。

### 4. 删除 `requirements.txt`

`pyproject.toml` + `uv.lock` 完全取代它。

### 5. 更新 `.gitignore`

现有内容只有三行（且 `playground.ipynn` 是拼写错误的历史遗留，保留不动）。追加：

```
__pycache__/
*.pyc
```

`.venv` 已在忽略列表中；`uv.lock`、`.python-version`、`pyproject.toml` **都要提交**（锁文件是可复现性的核心，不要忽略）。

顺带：`tools/__pycache__/*.pyc` 目前是被 git 跟踪的，加了忽略规则后应执行 `git rm -r --cached tools/__pycache__` 把它们从索引中移除。

### 6. 更新 `README.md` 的 Preparation / usage 段落

把第 16-39 行的 pip 流程换成 uv：

```markdown
## Preparation
1. install uv (if you haven't): https://docs.astral.sh/uv/getting-started/installation/
2. clone the repo
   `git clone https://github.com/Kumoyyds/product_category_match.git`
3. `cd product_category_match`
4. `uv sync`   # creates .venv and installs the exact locked dependencies
```

usage 里 `python main.py` 改为 `uv run main.py`（无需手动激活 venv）；notebook 用 `uv run --group dev jupyter ...` 或在 IDE 里选 `.venv` 解释器。同时修掉原文 `cd product_category_matching` 的目录名笔误。

## Verification

```powershell
uv sync                          # 应成功创建 .venv 并生成 uv.lock
uv run python -c "import pandas, numpy, joblib, yaml, openpyxl, dotenv, openai; print('deps ok')"
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 预期 2.8.x False（CPU wheel）
uv run python -c "from tools import trans; print('trans ok')"
uv run main.py
```

**已知的、与本次改动无关的前置问题**：`model/all-mpnet-base-v2` 在 git 里是一个 gitlink（模式 160000，指向 commit `e8c3b32`），但仓库根目录**没有 `.gitmodules`**，所以本地这个目录是空的。[tools/match_func.py:6-7](tools/match_func.py#L6-L7) 会在 import 时就 `SentenceTransformer(model_path)`，模型缺失时 `uv run main.py` 必然在这一步报错——这跟环境管理无关，需要另行把模型下载到该目录（或改成从 HuggingFace 拉 `sentence-transformers/all-mpnet-base-v2`）。因此把 `import tools.trans` / `import torch` 的检查作为本次改动是否成功的判据；`uv run main.py` 能跑到"loading match funcs"之后即说明依赖装对了。

前 4 条检查全过，即视为环境迁移完成。
