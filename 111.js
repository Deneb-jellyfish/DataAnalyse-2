const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageNumber
} = require('docx');
const fs = require('fs');

// ── colour palette ──────────────────────────────────────────────────────────
const BLUE      = "1F4E79";
const BLUE_LIGHT= "2E75B6";
const BLUE_BG   = "DEEAF1";
const GREEN     = "375623";
const GREEN_BG  = "E2EFDA";
const ORANGE    = "833C00";
const ORANGE_BG = "FCE4D6";
const PURPLE    = "3E1F6B";
const PURPLE_BG = "EAE0F5";
const GREY_BG   = "F2F2F2";
const WHITE     = "FFFFFF";

// ── border helpers ───────────────────────────────────────────────────────────
const thinBorder  = c => ({ style: BorderStyle.SINGLE, size: 1, color: c || "CCCCCC" });
const thickBorder = c => ({ style: BorderStyle.SINGLE, size: 4, color: c || "2E75B6" });
const allBorders  = (c) => { const b = thinBorder(c); return { top:b, bottom:b, left:b, right:b }; };
const noBorders   = () => {
  const n = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  return { top:n, bottom:n, left:n, right:n };
};

// ── paragraph helpers ────────────────────────────────────────────────────────
const p = (text, opts = {}) => new Paragraph({
  spacing: { before: opts.before ?? 60, after: opts.after ?? 60 },
  alignment: opts.align ?? AlignmentType.LEFT,
  children: [new TextRun({ text, size: opts.size ?? 20, font: "Arial",
    bold: opts.bold, color: opts.color, italics: opts.italics })]
});

const noteBox = (text) => new Table({
  width: { size: 9026, type: WidthType.DXA },
  columnWidths: [9026],
  rows: [new TableRow({ children: [new TableCell({
    borders: allBorders("2E75B6"),
    width: { size: 9026, type: WidthType.DXA },
    shading: { fill: BLUE_BG, type: ShadingType.CLEAR },
    margins: { top: 100, bottom: 100, left: 160, right: 160 },
    children: [new Paragraph({
      spacing: { before: 40, after: 40 },
      children: [new TextRun({ text, size: 19, font: "Arial", italics: true, color: BLUE })]
    })]
  })]})],
});

const space = (n = 120) => new Paragraph({ spacing: { before: 0, after: n }, children: [] });

// ── section heading ──────────────────────────────────────────────────────────
const h1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 320, after: 160 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE_LIGHT, space: 6 } },
  children: [new TextRun({ text, bold: true, size: 28, font: "Arial", color: BLUE })]
});

const h2 = (text, color) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 200, after: 100 },
  children: [new TextRun({ text, bold: true, size: 24, font: "Arial", color: color ?? BLUE })]
});

// ── role banner ──────────────────────────────────────────────────────────────
const roleBanner = (letter, title, subtitle, bgColor, textColor) => new Table({
  width: { size: 9026, type: WidthType.DXA },
  columnWidths: [1200, 7826],
  rows: [new TableRow({ children: [
    new TableCell({
      borders: noBorders(),
      width: { size: 1200, type: WidthType.DXA },
      shading: { fill: bgColor, type: ShadingType.CLEAR },
      margins: { top: 120, bottom: 120, left: 160, right: 160 },
      verticalAlign: "center",
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: letter, bold: true, size: 52, font: "Arial", color: WHITE })]
      })]
    }),
    new TableCell({
      borders: noBorders(),
      width: { size: 7826, type: WidthType.DXA },
      shading: { fill: bgColor, type: ShadingType.CLEAR },
      margins: { top: 100, bottom: 100, left: 200, right: 160 },
      children: [
        new Paragraph({ spacing: { before: 40, after: 20 },
          children: [new TextRun({ text: title, bold: true, size: 26, font: "Arial", color: WHITE })] }),
        new Paragraph({ spacing: { before: 0, after: 40 },
          children: [new TextRun({ text: subtitle, size: 20, font: "Arial", color: WHITE })] }),
      ]
    })
  ]})]
});

// ── day card table ───────────────────────────────────────────────────────────
const dayCard = (entries, accentColor, bgColor) => {
  // entries = [{ day, date, title, bullets, deliverable }]
  const cols = entries.length;
  const colW = Math.floor(9026 / cols);
  const colWidths = entries.map((_, i) => i === cols-1 ? 9026 - colW*(cols-1) : colW);

  const headerRow = new TableRow({ children: entries.map((e, i) => new TableCell({
    borders: allBorders(accentColor),
    width: { size: colWidths[i], type: WidthType.DXA },
    shading: { fill: accentColor, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 140, right: 140 },
    children: [
      new Paragraph({ spacing: { before: 20, after: 10 },
        children: [new TextRun({ text: e.day, bold: true, size: 20, font: "Arial", color: WHITE })] }),
      new Paragraph({ spacing: { before: 0, after: 20 },
        children: [new TextRun({ text: e.date, size: 18, font: "Arial", color: WHITE })] }),
      new Paragraph({ spacing: { before: 0, after: 20 },
        children: [new TextRun({ text: e.title, bold: true, size: 20, font: "Arial", color: WHITE })] }),
    ]
  })) });

  const maxBullets = Math.max(...entries.map(e => e.bullets.length));
  const bodyRows = [];
  for (let b = 0; b < maxBullets; b++) {
    bodyRows.push(new TableRow({ children: entries.map((e, i) => new TableCell({
      borders: allBorders(accentColor),
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { fill: bgColor, type: ShadingType.CLEAR },
      margins: { top: 60, bottom: 60, left: 140, right: 140 },
      children: [new Paragraph({
        spacing: { before: 30, after: 30 },
        children: e.bullets[b]
          ? [new TextRun({ text: "· " + e.bullets[b], size: 19, font: "Arial" })]
          : [new TextRun({ text: "", size: 19, font: "Arial" })]
      })]
    })) }));
  }

  const deliverableRow = new TableRow({ children: entries.map((e, i) => new TableCell({
    borders: allBorders(accentColor),
    width: { size: colWidths[i], type: WidthType.DXA },
    shading: { fill: GREY_BG, type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 140, right: 140 },
    children: [
      new Paragraph({ spacing: { before: 20, after: 10 },
        children: [new TextRun({ text: "交付：", bold: true, size: 18, font: "Arial", color: accentColor })] }),
      new Paragraph({ spacing: { before: 0, after: 20 },
        children: [new TextRun({ text: e.deliverable, size: 18, font: "Arial", italics: true })] }),
    ]
  })) });

  return new Table({
    width: { size: 9026, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...bodyRows, deliverableRow]
  });
};

// ── standard table helper ────────────────────────────────────────────────────
const stdTable = (headers, rows, colWidths, accentColor) => {
  const headerRow = new TableRow({ children: headers.map((h, i) => new TableCell({
    borders: allBorders(accentColor),
    width: { size: colWidths[i], type: WidthType.DXA },
    shading: { fill: accentColor, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 140, right: 140 },
    children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: h, bold: true, size: 20, font: "Arial", color: WHITE })] })]
  })) });
  const dataRows = rows.map((row, ri) => new TableRow({ children: row.map((cell, ci) => new TableCell({
    borders: allBorders("AAAAAA"),
    width: { size: colWidths[ci], type: WidthType.DXA },
    shading: { fill: ri % 2 === 0 ? WHITE : GREY_BG, type: ShadingType.CLEAR },
    margins: { top: 70, bottom: 70, left: 140, right: 140 },
    children: [new Paragraph({ children: [new TextRun({ text: cell, size: 19, font: "Arial" })] })]
  })) }));
  return new Table({
    width: { size: 9026, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows]
  });
};

// ════════════════════════════════════════════════════════════════════════════
// DOCUMENT
// ════════════════════════════════════════════════════════════════════════════
const doc = new Document({
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 560, hanging: 280 } } } }] }
    ]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1260, right: 1260, bottom: 1260, left: 1260 }
      }
    },
    children: [

      // ── COVER ──────────────────────────────────────────────────────────────
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 1200, after: 200 },
        children: [new TextRun({ text: "团队详细分工说明书", bold: true, size: 52, font: "Arial", color: BLUE })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 120 },
        children: [new TextRun({ text: "多城市空气质量（AQI）预测算法系统性对比研究", bold: true, size: 32, font: "Arial", color: BLUE_LIGHT })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 80 },
        children: [new TextRun({ text: "数据挖掘期末项目 · 选项B · 三人小组", size: 22, font: "Arial", color: "888888" })]
      }),
      noteBox("本文档面向全体组员。读完后每人应清楚：自己在哪个阶段做什么、产出什么文件、依赖谁、被谁依赖。"),
      space(200),

      // ── CH1 角色总览 ────────────────────────────────────────────────────────
      h1("第一章  角色分配总览"),
      space(80),
      stdTable(
        ["组员", "角色", "核心职责范围", "报告章节", "工作量占比"],
        [
          ["组员A", "数据工程负责人", "数据获取、清洗、特征工程、EDA可视化、数据血缘图", "第2章（数据描述与预处理）", "约 33%"],
          ["组员B", "算法负责人（兼评估分析）", "5种模型实现与调优、评估框架、统计检验、误差分析", "第3章（方法）、第4章（实验）", "约 34%"],
          ["组员C", "工程与交付负责人（兼可视化）", "Pipeline整合、Streamlit界面、可视化图表、README、PPT、视频", "第1章（摘要/引言）、第6章（结论）", "约 33%"],
        ],
        [900, 2000, 3226, 1800, 1100],
        BLUE_LIGHT
      ),
      space(80),
      p("说明：本组3人，在原始4角色基础上合并调整。组员B同时承担算法与评估，组员C同时承担工程交付与可视化。报告第5章讨论由三人共同撰写。GitHub commit记录作为工作量量化依据，每人不少于10条。", { color: "666666", italics: true }),
      space(160),

      // ── CH2 组员A ───────────────────────────────────────────────────────────
      h1("第二章  组员A — 数据工程负责人"),
      space(60),
      roleBanner("A", "组员A　数据工程负责人", "负责：数据获取 · 清洗 · 特征工程 · EDA · 数据血缘图", "2E75B6", WHITE),
      space(120),

      h2("阶段一：数据获取与探索（第14周）"),
      p("从 Kaggle 下载 Beijing Multi-Site Air-Quality Data，同步获取上海同期 AQI 数据用于迁移实验。完成探索性数据分析（EDA），绘制以下图表："),
      space(40),
      dayCard([
        {
          day: "任务 A-1",
          date: "第14周",
          title: "数据下载与初步探查",
          bullets: [
            "下载北京12站点AQI数据集（约43万条）",
            "下载上海同期AQI数据（迁移实验备用）",
            "读取数据，检查字段名、数据类型、行数",
            "统计各字段缺失率，输出缺失值热力图",
          ],
          deliverable: "data/raw/ 目录 + notebooks/01_EDA.ipynb 骨架"
        },
        {
          day: "任务 A-2",
          date: "第14周末",
          title: "EDA可视化（4张图）",
          bullets: [
            "PM2.5/AQI时序折线图（全年趋势）",
            "偶极矩/AQI分布直方图（确定预处理方案）",
            "原子/污染物类型占比图",
            "各特征相关性热力图（19个字段）",
            "每张图附2~3句文字说明",
          ],
          deliverable: "4张PNG + notebooks/01_EDA.ipynb 完成版"
        }
      ], "2E75B6", BLUE_BG),
      space(120),

      h2("阶段二：数据清洗（第15周上半段）"),
      space(40),
      dayCard([
        {
          day: "任务 A-3",
          date: "第15周周一~周二",
          title: "缺失值与异常值处理",
          bullets: [
            "线性插值填充连续缺失≤6小时的样本",
            "超过6小时的连续缺失：标记后剔除",
            "基于3σ原则识别传感器故障造成的尖刺值",
            "记录处理前后数据量变化，写入文档",
          ],
          deliverable: "utils/preprocess.py（清洗函数）"
        },
        {
          day: "任务 A-4",
          date: "第15周周三",
          title: "时间聚合与城市对齐",
          bullets: [
            "将12个站点小时数据聚合为城市日均值",
            "北京/上海数据时间索引对齐",
            "确保两城市字段名、单位一致",
            "输出 data/processed/beijing.csv 和 shanghai.csv",
          ],
          deliverable: "data/processed/ 目录 + 聚合脚本"
        }
      ], "2E75B6", BLUE_BG),
      space(120),

      h2("阶段三：特征工程（第15周，XGBoost专用）"),
      space(40),
      dayCard([
        {
          day: "任务 A-5",
          date: "第15周周四",
          title: "时序滞后特征",
          bullets: [
            "lag-1（昨日均值）",
            "lag-3（三天前）",
            "lag-7（上周同期）",
            "7日/14日滚动均值",
            "7日/14日滚动标准差",
          ],
          deliverable: "utils/feature_engineering.py（函数）"
        },
        {
          day: "任务 A-6",
          date: "第15周周五",
          title: "时间与交互特征",
          bullets: [
            "月份正弦/余弦周期编码",
            "星期几编码",
            "是否节假日（0/1）",
            "温度×湿度交互项",
            "风速×风向分量",
            "输出X_train.npy、X_test.npy，传给组员B",
          ],
          deliverable: "特征矩阵.npy文件 + docs/feature_doc.md"
        }
      ], "2E75B6", BLUE_BG),
      space(120),

      h2("阶段四：数据血缘图与报告（第16周）"),
      p("绘制数据血缘图，清晰展示从原始数据到各模型输入的完整处理链路。负责撰写报告第2章（数据描述与预处理）。"),
      space(80),

      noteBox("组员A在第15周周五前须将 data/processed/ 和特征矩阵文件推送到GitHub，组员B依赖这份数据才能开始训练。这是整个项目最关键的交接节点。"),
      space(80),

      h2("组员A可考核输出文件"),
      space(40),
      stdTable(
        ["文件/目录", "说明"],
        [
          ["data/raw/", "原始数据集（不修改）"],
          ["data/processed/", "清洗后北京/上海日均值CSV"],
          ["utils/preprocess.py", "缺失值处理、聚合函数"],
          ["utils/feature_engineering.py", "特征构造函数（XGBoost专用）"],
          ["notebooks/01_EDA.ipynb", "探索性分析，含6张可视化图"],
          ["docs/data_lineage.png", "数据血缘图"],
          ["docs/feature_doc.md", "特征说明文档（字段名、含义、取值范围）"],
          ["报告第2章", "数据描述、预处理步骤、EDA结论"],
        ],
        [3000, 6026],
        BLUE_LIGHT
      ),
      space(200),

      // ── CH3 组员B ───────────────────────────────────────────────────────────
      h1("第三章  组员B — 算法负责人（兼评估分析）"),
      space(60),
      roleBanner("B", "组员B　算法负责人（兼评估分析）", "负责：5种模型实现与调优 · 评估框架 · 统计显著性检验 · 误差分析", "375623", WHITE),
      space(120),

      h2("阶段一：基线模型（第15周上半段）"),
      space(40),
      dayCard([
        {
          day: "任务 B-1",
          date: "第15周周一~周二",
          title: "ARIMA",
          bullets: [
            "使用 auto_arima 自动定阶",
            "输出最优 (p,d,q) 参数",
            "在测试集做初步预测，验证数据管道可用",
            "记录训练时间和初步RMSE",
          ],
          deliverable: "models/arima_model.py"
        },
        {
          day: "任务 B-2",
          date: "第15周周二~周三",
          title: "Prophet",
          bullets: [
            "配置年/周季节性参数",
            "加入中国法定节假日",
            "输出预测值与置信区间",
            "记录训练时间和初步RMSE",
          ],
          deliverable: "models/prophet_model.py"
        }
      ], "375623", GREEN_BG),
      space(120),

      h2("阶段二：进阶模型（第15周下半段）"),
      space(40),
      dayCard([
        {
          day: "任务 B-3",
          date: "第15周周三~周四",
          title: "XGBoost（特征工程版）",
          bullets: [
            "对接组员A的特征矩阵（X_train.npy）",
            "使用时间序列交叉验证调参",
            "调优：learning_rate、max_depth、n_estimators",
            "贝叶斯优化寻找最优超参",
          ],
          deliverable: "models/xgboost_model.py"
        },
        {
          day: "任务 B-4",
          date: "第15周周四~周五",
          title: "LSTM",
          bullets: [
            "设计输入窗口 lookback=30天",
            "隐藏层：2层LSTM + Dropout",
            "加入 Early Stopping 防止过拟合",
            "调优：learning_rate、batch_size、window",
          ],
          deliverable: "models/lstm_model.py"
        }
      ], "375623", GREEN_BG),
      space(80),
      dayCard([
        {
          day: "任务 B-5",
          date: "第15周周末",
          title: "Transformer / Informer",
          bullets: [
            "使用开源 Informer 实现，适配数据格式",
            "调整 seq_len、label_len、pred_len",
            "验证长序列预测效果",
            "记录训练时间（与LSTM对比）",
          ],
          deliverable: "models/transformer_model.py"
        },
        {
          day: "任务 B-6",
          date: "第15周末",
          title: "超参数调优汇总",
          bullets: [
            "ARIMA：grid search (p,d,q)",
            "XGBoost：贝叶斯优化",
            "LSTM/Transformer：消融实验（window size）",
            "每次实验配置+结果记录进实验记录表",
          ],
          deliverable: "docs/experiment_log.csv（参数+指标）"
        }
      ], "375623", GREEN_BG),
      space(120),

      h2("阶段三：评估框架（第15周末~第16周初）"),
      space(40),
      dayCard([
        {
          day: "任务 B-7",
          date: "第16周周一",
          title: "统一评估接口",
          bullets: [
            "输入任意模型预测序列，输出 MAE/RMSE/MAPE",
            "按 AQI 等级分层（优/良/轻度/重度）分别计算",
            "生成5模型横向对比汇总表",
            "输出 predictions.csv 传给组员C画图",
          ],
          deliverable: "evaluation/metrics.py + predictions.csv"
        },
        {
          day: "任务 B-8",
          date: "第16周周二",
          title: "显著性检验 + 迁移评估",
          bullets: [
            "对最优两个模型做 Diebold-Mariano 检验",
            "输出检验统计量和 p-value",
            "将北京最优模型直接在上海测试集推理",
            "记录跨城市泛化误差，分析差异原因",
          ],
          deliverable: "evaluation/dm_test.py + 迁移结果表格"
        }
      ], "375623", GREEN_BG),
      space(120),

      noteBox("组员B在第16周周二前须将 predictions.csv 推送到 GitHub，组员C依赖这份文件才能画散点图和误差对比图。"),
      space(80),

      h2("组员B可考核输出文件"),
      space(40),
      stdTable(
        ["文件/目录", "说明"],
        [
          ["models/arima_model.py", "ARIMA实现，含自动定阶"],
          ["models/prophet_model.py", "Prophet实现，含节假日配置"],
          ["models/xgboost_model.py", "XGBoost实现，含特征对接"],
          ["models/lstm_model.py", "LSTM实现，含Early Stopping"],
          ["models/transformer_model.py", "Transformer/Informer实现"],
          ["evaluation/metrics.py", "统一评估接口（MAE/RMSE/MAPE + 分层）"],
          ["evaluation/dm_test.py", "Diebold-Mariano显著性检验"],
          ["docs/experiment_log.csv", "超参数调优实验记录表"],
          ["predictions.csv", "全部模型在测试集的预测值（传给组员C）"],
          ["报告第3章", "5种算法原理简述"],
          ["报告第4章", "实验设置、评估结果表格、DM检验结论"],
        ],
        [3000, 6026],
        "375623"
      ),
      space(200),

      // ── CH4 组员C ───────────────────────────────────────────────────────────
      h1("第四章  组员C — 工程与交付负责人（兼可视化）"),
      space(60),
      roleBanner("C", "组员C　工程与交付负责人（兼可视化）", "负责：Pipeline整合 · Streamlit界面 · 可视化 · README · PPT · 演示视频", "833C00", WHITE),
      space(120),

      h2("阶段一：工程基础搭建（第14周，与组员A并行）"),
      space(40),
      dayCard([
        {
          day: "任务 C-1",
          date: "第14周",
          title: "仓库与规范初始化",
          bullets: [
            "创建 GitHub 仓库，配置 .gitignore",
            "建立项目目录结构（见下方）",
            "统一代码规范：命名、注释风格",
            "编写 requirements.txt 初稿",
          ],
          deliverable: "GitHub仓库 + 目录结构 + requirements.txt"
        },
        {
          day: "任务 C-2",
          date: "第14周末",
          title: "Pipeline骨架搭建",
          bullets: [
            "编写 main.py 框架，预留各模块调用接口",
            "定义各模块输入输出数据格式（约定好字段名）",
            "确保骨架可运行（各步骤打印占位符）",
            "与组员A/B确认接口格式",
          ],
          deliverable: "main.py骨架 + 接口约定文档"
        }
      ], "833C00", ORANGE_BG),
      space(120),

      h2("阶段二：可视化模块（第15~16周，随模型完成逐步推进）"),
      p("所有图表函数化，封装进 visualization/plot.py，同时被 main.py 和 Streamlit 调用。"),
      space(40),
      dayCard([
        {
          day: "任务 C-3",
          date: "第15周末（拿到predictions.csv后）",
          title: "预测结果图表",
          bullets: [
            "预测曲线图：真实值 vs 5模型预测值叠加",
            "含Prophet置信区间阴影",
            "误差对比柱状图：MAE/RMSE/MAPE横向对比",
            "最优模型散点图：预测值 vs 真实值",
          ],
          deliverable: "visualization/plot.py（图表函数）"
        },
        {
          day: "任务 C-4",
          date: "第16周初",
          title: "分析与评估图表",
          bullets: [
            "分层误差热力图：AQI等级 × 模型 误差矩阵",
            "训练过程 loss 曲线（LSTM/Transformer）",
            "多城市对比折线图：北京/上海预测效果",
            "特征重要性图：XGBoost各特征贡献度条形图",
          ],
          deliverable: "outputs/ 目录中全部PNG图（共6~8张）"
        }
      ], "833C00", ORANGE_BG),
      space(120),

      h2("阶段三：Streamlit 交互界面（第15周末—第16周初）"),
      space(40),
      dayCard([
        {
          day: "任务 C-5",
          date: "第16周周一~周二",
          title: "界面布局与基础功能",
          bullets: [
            "侧边栏：城市选择、时间范围、模型多选",
            "主区域：预测曲线对比图（动态渲染）",
            "评估指标汇总表（可交互排序）",
            "连接 predictions.csv 和可视化函数",
          ],
          deliverable: "app.py（Streamlit入口）基础版"
        },
        {
          day: "任务 C-6",
          date: "第16周周三",
          title: "进阶功能与美化",
          bullets: [
            "AQI等级分层误差分析标签页",
            "多城市迁移结果展示",
            "页面配色、字体统一",
            "确保在本地 streamlit run app.py 可正常运行",
          ],
          deliverable: "app.py 完整版"
        }
      ], "833C00", ORANGE_BG),
      space(120),

      h2("阶段四：Pipeline整合（第16周）"),
      space(40),
      dayCard([
        {
          day: "任务 C-7",
          date: "第16周周三",
          title: "一键运行Pipeline",
          bullets: [
            "将预处理、5模型训练、评估、出图串联进 main.py",
            "解决各模块接口不一致问题",
            "python main.py 可从原始数据跑完全流程",
            "测试 python main.py 在干净环境下可运行",
          ],
          deliverable: "main.py 完整版（可一键运行）"
        },
        {
          day: "任务 C-8",
          date: "第16周周四",
          title: "README与requirements",
          bullets: [
            "README：环境安装、数据准备、运行方式、目录说明",
            "requirements.txt：锁定所有依赖版本号",
            "验证克隆仓库后按README可成功复现结果",
          ],
          deliverable: "README.md + requirements.txt 最终版"
        }
      ], "833C00", ORANGE_BG),
      space(120),

      h2("阶段五：交付物制作（第16周末）"),
      space(40),
      dayCard([
        {
          day: "任务 C-9",
          date: "第16周周五",
          title: "答辩PPT（12~15页）",
          bullets: [
            "封面、问题背景（1页）",
            "数据介绍与EDA（2页）",
            "5种算法原理简述（2页）",
            "实验结果与对比表（3页）",
            "多城市迁移实验（1页）",
            "结论与展望（1页）",
          ],
          deliverable: "slides.pptx"
        },
        {
          day: "任务 C-10",
          date: "第16周周末",
          title: "演示视频（约5分钟）",
          bullets: [
            "0~1分钟：背景与数据介绍",
            "1~3分钟：5模型对比结果展示",
            "3~5分钟：Streamlit实时演示",
            "报告第1章（摘要/引言）",
            "报告第6章（结论）",
          ],
          deliverable: "demo_video.mp4 + 报告第1、6章"
        }
      ], "833C00", ORANGE_BG),
      space(120),

      noteBox("组员C负责在第16周周三组织三人联合跑通 python main.py 全流程。这是提交前的关键里程碑，任何接口问题必须在这一步解决。"),
      space(80),

      h2("组员C可考核输出文件"),
      space(40),
      stdTable(
        ["文件/目录", "说明"],
        [
          ["main.py", "一键运行主流程（预处理→训练→评估→出图）"],
          ["app.py", "Streamlit交互界面入口"],
          ["visualization/plot.py", "全部可视化函数（被main.py和app.py复用）"],
          ["outputs/", "全部PNG图表（6~8张）"],
          ["README.md", "项目说明文档（安装、运行、目录结构）"],
          ["requirements.txt", "依赖列表（版本锁定）"],
          ["slides.pptx", "答辩PPT（12~15页）"],
          ["demo_video.mp4", "演示视频（约5分钟）"],
          ["报告第1章", "摘要、引言"],
          ["报告第6章", "结论与展望"],
        ],
        [3000, 6026],
        "833C00"
      ),
      space(200),

      // ── CH5 协作约定 ────────────────────────────────────────────────────────
      h1("第五章  协作约定与时间节点"),
      space(60),

      h2("5.1 任务依赖关系"),
      space(40),
      stdTable(
        ["时间节点", "事件", "阻塞谁"],
        [
          ["第14周末", "组员C完成仓库结构和Pipeline骨架", "三人代码都依赖统一目录结构"],
          ["第15周周五", "组员A提交 data/processed/ 和特征矩阵到GitHub", "组员B无法开始XGBoost/LSTM训练"],
          ["第15周末", "组员B完成全部5个模型", "组员C无法开始可视化对接"],
          ["第16周周二", "组员B提交 predictions.csv 到GitHub", "组员C无法画散点图和误差对比图"],
          ["第16周周三", "三人联合跑通 python main.py 全流程", "提交前必须完成，不能留到最后"],
          ["第16周周四", "三人各自完成负责的报告章节初稿", "组员C汇总统一格式需要所有章节"],
          ["第16周末", "全部交付物提交", "最终截止"],
        ],
        [1600, 4426, 3000],
        BLUE_LIGHT
      ),
      space(120),

      h2("5.2 最容易卡壳的地方"),
      space(40),
      stdTable(
        ["风险点", "谁受影响", "预防方案"],
        [
          ["PyTorch/torch-geometric版本冲突", "组员B", "第14周就装好，装不上立即在群里说，不要拖到第15周"],
          ["上海AQI数据下载失败（网络问题）", "组员A", "优先从 Kaggle 手动下载，备用方案：只用北京数据完成主体"],
          ["LSTM训练loss不下降", "组员B", "先检查学习率（换0.0001），再检查数据归一化是否正确"],
          ["Streamlit界面跑不起来（依赖缺失）", "组员C", "requirements.txt 锁定版本，README写清楚安装步骤"],
          ["各模块接口对接失败（字段名不一致）", "组员C", "第14周就约定好数据格式，写进接口约定文档"],
          ["报告超过20页", "全体", "每章负责人严格控制篇幅，第5章讨论控制在1.5页以内"],
        ],
        [2400, 1600, 5026],
        BLUE_LIGHT
      ),
      space(120),

      h2("5.3 代码规范约定"),
      space(40),
      p("所有人遵守以下约定，减少合并冲突："),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { before: 40, after: 40 },
        children: [new TextRun({ text: "统一用 Python 3.10+，不使用 walrus operator（:=）以外的3.10+专用语法", size: 20, font: "Arial" })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { before: 40, after: 40 },
        children: [new TextRun({ text: "每个函数写 docstring，说明输入输出格式", size: 20, font: "Arial" })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { before: 40, after: 40 },
        children: [new TextRun({ text: "禁止把数据文件（.csv/.npy）推送到 GitHub，用 .gitignore 排除 data/ 目录", size: 20, font: "Arial" })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { before: 40, after: 40 },
        children: [new TextRun({ text: "commit message 格式：[A/B/C] 简短描述，如 [B] add LSTM model with early stopping", size: 20, font: "Arial" })] }),
      new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { before: 40, after: 40 },
        children: [new TextRun({ text: "不在 main 分支直接 push，各自在 feature 分支开发，完成后 PR 合并", size: 20, font: "Arial" })] }),
      space(120),

      h2("5.4 修订后项目时间表"),
      space(40),
      stdTable(
        ["节点", "截止时间", "内容", "负责人"],
        [
          ["T1", "第14周末", "提交计划书；EDA完成；仓库结构建好；Pipeline骨架完成", "全体 / A主数据、C主工程"],
          ["T2", "第15周周五", "数据清洗特征工程完成；ARIMA、Prophet完成", "A（数据）/ B（模型）"],
          ["T3", "第15周末", "XGBoost、LSTM、Transformer完成；评估框架搭好", "B主导 / A协助特征"],
          ["T4", "第16周周三", "可视化完成；Streamlit完成；main.py联调通过；报告初稿各章完成", "C主导 / 全体参与报告"],
          ["T5", "第16周末", "报告定稿；PPT完成；视频录制；全部提交", "全体"],
        ],
        [600, 1200, 4826, 2400],
        BLUE_LIGHT
      ),
      space(120),

      noteBox("原计划书将数据清洗、5个模型训练、多城市实验全部压缩在第15周周二~周六（约5天），执行风险极高。以上时间表已调整为以周为单位，并在第16周保留联调和报告缓冲时间。"),

    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("./详细分工说明书_AQI项目.docx", buffer);
  console.log("Done");
});