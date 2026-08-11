import { defineConfig } from 'vitepress'
import { katex } from '@mdit/plugin-katex'

// 站点侧边栏 = 课程的地图。
// 分组方式刻意按「MLSys / AI Infra 体系的认知顺序」组织，
// 而不是课程周次的平铺顺序：先问题，再硬件，再软件栈，再大模型系统。
export default defineConfig({
  lang: 'zh-CN',
  title: 'MLSys 中文整理',
  description:
    '复旦大学《机器学习系统》课程教材的中文整理：从 GPU 架构、CUDA 到深度学习编译器、大模型训练与推理系统。',

  base: '/fudan-mlsys-cn/',
  cleanUrls: true,
  lastUpdated: true,

  head: [
    ['meta', { name: 'theme-color', content: '#0b6bcb' }],
    // KaTeX 样式：随包版本本地引入（node_modules/katex/dist/katex.min.css 拷贝到 docs/public/katex.min.css），避免 CDN 版本漂移
    ['link', { rel: 'stylesheet', href: '/fudan-mlsys-cn/katex.min.css' }],
  ],

  markdown: {
    config: (md) => {
      md.use(katex, { output: 'html' })
    },
  },

  themeConfig: {
    logo: 'https://memxlife.github.io/favicon.ico',

    nav: [
      { text: '课程主页', link: 'https://memxlife.github.io/courses/mlsys.html' },
      { text: 'GitHub', link: 'https://github.com/Meredith2328/fudan-mlsys-cn' },
    ],

    sidebar: [
      {
        text: '第一篇 · 为什么需要 MLSys',
        collapsed: false,
        items: [
          { text: '第 1 章 机器学习系统：问题、约束与垂直整合', link: '/chapter1' },
          { text: '第 2 章 CPU 基础、GPU 的兴起与吞吐计算的逻辑', link: '/chapter2' },
        ],
      },
      {
        text: '第二篇 · GPU 架构与 CUDA',
        collapsed: false,
        items: [
          { text: '第 3 章 GPU 架构与机器学习系统', link: '/chapter3' },
          { text: '第 4 章 CUDA 编程从架构开始', link: '/chapter4' },
          { text: '第 5 章 CUDA 编程作为软硬件协同优化', link: '/chapter5' },
        ],
      },
      {
        text: '第三篇 · 智能体运行时与深度学习编译器',
        collapsed: false,
        items: [
          { text: '第 6 章 Claude Code 内部机制：重建与解读', link: '/chapter6' },
          { text: '第 7 章 深度学习编译器：从通用性到专用化，再回到学习式搜索', link: '/chapter7' },
        ],
      },
      {
        text: '第四篇 · 大模型训练与推理系统',
        collapsed: false,
        items: [
          { text: '第 8 章 Transformer 基础', link: '/chapter8' },
          { text: '第 9 章 分布式训练与数据并行', link: '/chapter9' },
          { text: '第 10 章 模型并行与专家并行', link: '/chapter10' },
          { text: '第 11 章 LLM 推理系统', link: '/chapter11' },
          { text: '第 12 章 分布式训练基础设施', link: '/chapter12' },
        ],
      },
      {
        text: '专题',
        items: [
          { text: 'DeepSeek 技术路线图：效率护城河', link: '/deepseek' },
        ],
      },
      {
        text: '课程项目',
        items: [
          { text: '项目 1 GPU 性能分析与硬件探测', link: '/project1' },
          { text: '项目 2 LoRA 算子的 Agentic 优化', link: '/project2' },
          { text: '项目 3 自动化 LLM 推理运行时', link: '/project3' },
        ],
      },
      { text: '附录 · Profiling 教程', link: '/tutorial' },
    ],

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            displayDetails: '显示详细列表',
            resetButtonTitle: '清除查询',
            noResultsText: '没有找到结果：',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },

    outline: { label: '本页导航', level: [2, 3] },
    docFooter: { prev: '上一篇', next: '下一篇' },
    lastUpdated: { text: '更新于', formatOptions: { dateStyle: 'short', timeStyle: 'short' } },
    notFound: {
      title: '页面不存在',
      quote: '要么链接写错了，要么这一章还没写完。',
      linkText: '回到首页',
    },
  },
})
