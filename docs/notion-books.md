# 爬取后的输出：EPUB、Notion 草稿与 TXT

新增书籍时，先根据用户要求确定输出目的地；未指定时询问。可选择本地 EPUB、
[Notion 草稿](https://app.notion.com/p/3deca693996b810c8774f3658c89a423)、
TXT 或任意组合。爬取一次后写入所选目的地；单独选择本地文件时不连接 Notion。
Notion 输出完成于草稿写入和回读验证；发布操作不属于上传流程。

## 安装共享依赖

先安装 `uv`、`mise` 和 GitHub CLI（`gh`）。`notion-books` 是必装依赖，
即使仅输出本地 EPUB，也需要能访问其私有 GitHub 仓库的账号。
依赖固定到版本标签，`uv.lock` 记录确切提交；无需相邻 checkout。

首次安装在本项目根目录依次运行：

```bash
gh auth login
gh auth setup-git
mise trust
mise install
mise exec -- uv sync --locked
mise exec -- uv run --locked book-notion --help
```

最后一条命令显示帮助即表示依赖安装和 CLI 加载成功，不会连接 Notion。
Git 报权限错误时，确认 `gh` 登录的账号拥有 `notion-books` 仓库访问权限，
再运行 `gh auth setup-git` 并重试同步。Go 编译器缺失时，运行 `mise install`，
并通过 `mise exec --` 执行同步。Go 核心在安装时编译进 Python 包；运行时无需 Go 或新增服务。

测试共享库的本地修改时，使用临时覆盖，不修改依赖配置或锁文件：

```bash
mise exec -- uv run --with /absolute/path/to/notion-books python -m unittest discover -s tests -p 'test_*.py'
```

正式升级时选择已发布的不可变标签，同步修改本项目的版本号和 Git 标签，运行
`mise exec -- uv lock`。移除本地覆盖后，使用锁定版本运行本项目测试。

## 使用

```bash
uv run book-notion login
uv run book-ingest "作品URL" --mode notion
uv run book-ingest "作品URL" --mode epub -o books/作者/书名.epub
uv run book-ingest "作品URL" --mode epub --mode notion --mode txt
```

`book-ingest` 的 `--mode`（别名 `--output-format`）可重复；`book-to-epub`
使用可重复的 `--output-format`。未指定模式时，`-o` 选择 EPUB，
`--txt-output` 选择 TXT；都未指定时 CLI 报错，避免隐式上传。
`both` 保留为 EPUB + TXT 的别名。训练用途选择 TXT、数据集路径和对应的
manifest 更新，与普通抓书使用同一流程。组合输出逐项执行，失败时先核对
已有文件及上传检查点，再恢复未完成的输出。
翻译流程 `book-translate build-epub` 仍然输出本地文件。

正文、目录、作者关联和书籍属性通过独立 MCP 客户端读写。
OAuth 凭据保存在用户私有状态目录，可刷新；无需保持 Notion 网页登录。
`book-notion logout` 删除本地 token，重新授权使用 `book-notion login`。

### 封面

Notion 路径自动上传爬虫取得的封面，保存为书页的原生 cover。
MCP 的 cover 参数只接受外部 URL，因此仅封面使用官方 File Upload API。
在运行脚本的环境中设置 `NOTION_API_TOKEN`，并给该 integration 目标 Notion 页面的访问权限。
脚本不读取 `.env`，不保存 token 或上传后的临时下载 URL。
未提供封面时无需 API token；封面为 PNG/JPEG，最大 10 MiB、2500 万像素。
封面不可读取、上传失败或回读不一致时停止，保留上传检查点。

本地 EPUB 嵌入封面图片及 cover metadata，供书库显示缩略图；
不生成 `cover.xhtml` 或 `cover.html`，也不加入封面阅读页。

## Notion 存储

目录 ID 配置在 `book_specs/notion/config.json`。新书使用作品库默认模板创建：

- 每本书拥有独立的「正文」数据库。`章节`是页面标题，`所属标题`是可选的上级目录标题。
- 读取未筛选、未分组、无属性排序的「正文」手动视图；「待发布」视图不决定目录。
- 上传按源目录建立行顺序，完成后核对实际手动顺序。相同上级标题只有连续出现时才归为一组。
- Notion 上传支持章节加一层上级标题；更深的目录在上传前报错，本地 EPUB 路径仍支持原目录树。
- 简介、尾声、后记和番外卷中的章节都是正文行。简介元数据与可阅读的简介章节分别保存。
- 独立番外存放于共享库，通过「涉及作品」关联书籍。书页的番外视图只筛选当前作品，并保留手动顺序。

作品属性写入 `作品`、`作者` relation、`书籍分类` multi-select、`系列` select、
`系列序号` number、`语言` select、`简介`、`来源` URL、`出版日期` date。
可由源提供多个作者；不会自行拆分笔名。上传器只补齐必要的 select 选项，保留已有值。
上传只写上述内容属性，保留目标库中的其他属性和按钮。

## 续传与冲突

检查点位于 `generated/notion_cms_sources/<来源摘要>/import.json`，按目标目录及来源标识隔离。
保存每次创建返回的 ID，回读正文和属性后标记已验证；上传完成后以 Notion 中的编辑为准。

```bash
uv run book-notion resume --state generated/notion_cms_sources/<来源摘要>/import.json
```

- 创建请求结果不确定：根据页面内容核对对应 ID，再修复检查点；不要盲目重建。
- 同名书已存在：核对书籍身份及原检查点，不覆盖已有书籍。
- 新爬取与检查点不同：比较本地来源和 Notion 编辑，确认合并后再继续。
- 疑似重复番外：上传前暂停，在检查点同目录生成 `extra-review.md`。按报告选择
  复用已有正文或新建，再续传；操作与判定范围见[共享番外](fanwai-notion.md)。
- 实际顺序不同：在对应手动视图整理为源目录顺序，再续传；不会添加数字排序列。
- `cover_pending` 表示封面附加结果不确定。核对页面封面及原文件；确认成功后标记
  `cover_uploaded: true` 并移除 `cover_pending`，确认未附加才清除该标记重试。

不同目标库之间不能复用导入检查点。
已上传内容在 Notion 中编辑；重新抓取本地版则选择 `--mode epub`。

## 排版约定

`src/content/` 负责内容相关的清理和格式识别；共享依赖 `notion-books` 负责
Notion schema、格式编解码和读写。`src/notion/cms.py` 与 `upload.py`
负责导入决策、身份匹配、检查点和流程；认证与封面 HTTP 客户端也留在本项目。
`src/workflows/ingest.py` 决定输出目的地；本地输出调用 `src/epub/writer.py`，不经 Notion。
模块边界见[架构说明](architecture.md)。

正文和番外使用同一组块和渲染规则。保持现有 CSS：正文行高 `1.7`、段间距 `0.3em`，
正文内 H3 为 `1.1em`。标题级别和对齐独立；居中用三列中间列存内容、两侧留空，
右对齐用最右列。粗斜体、下划线、删除线、段内换行、空段、引用和分隔线保持语义。
渲染只解释排版属性，不根据「全文完」或其他具体文字推断格式。

本地替换先生成、验证候选 EPUB，再校验原文件哈希、备份和原子安装。
检查点、书籍内容和生成的 EPUB 不提交 Git。

## Live upload validation

The [crawler live test](../tests/LIVE_NOTION.md) runs a synthetic crawl result
through real Notion upload, readback and checkpoint resume. The test ends after
verifying the uploaded content and preserved page identities.
