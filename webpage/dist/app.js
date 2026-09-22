const esc = (value) => String(value).replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));

// Keep best/second-best annotations local to the marked value. Row shading
// identifies the SpeakerMem variant; it does not make every number optimal.
const bold = (value) => `<strong>${esc(value)}</strong>`;
const best = (value) => `<strong class="best">${esc(value)}</strong>`;
const second = (value) => `<u class="second">${esc(value)}</u>`;
const pair = (acc, f1, accMark = '', f1Mark = '') => `${accMark === 'best' ? best(acc) : accMark === 'second' ? second(acc) : esc(acc)} / ${f1Mark === 'best' ? best(f1) : f1Mark === 'second' ? second(f1) : esc(f1)}`;

function renderTable(target, columns, rows, options = {}) {
  const el = typeof target === 'string' ? document.querySelector(target) : target;
  if (!el) return;
  const head = columns.map((column) => `<th>${esc(column)}</th>`).join('');
  const body = rows.map((row) => {
    const values = row.cells || row;
    const rowClass = row.className ? ` class="${row.className}"` : '';
    return `<tr${rowClass}>${values.map((cell) => `<td>${cell}</td>`).join('')}</tr>`;
  }).join('');
  el.innerHTML = `<table class="paper-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

const mainSets = {
  deepseek: {
    caption: 'Main benchmark results with DeepSeek-V4-Flash. Paired cells report Acc. / token-F1; MeanQ and MeanN are SocialMemBench summary scores.',
    note: 'SpeakerMem-R1 reaches 47.9% on GroupMemBench, 64.9% on SocialMemBench, and 61.9% on EverMemBench. Full context is feasible only on SocialMemBench; the other histories cannot reliably fit within the configured context window.',
    rows: [
      ['DeepSeek-V4-Flash','BM25',pair('44.6','25.0'),pair('28.6','15.7'),'.234','.250',pair('64.4','3.6'),pair('27.0','23.3','','best'),pair('52.5','9.9')],
      ['DeepSeek-V4-Flash','Embed',pair('34.5','17.4'),pair('38.1','18.1'),'.300','.307',pair('56.1','4.0'),pair('23.9','22.7','','second'),pair('45.9','10.0')],
      ['DeepSeek-V4-Flash','Mem0',pair('21.6','9.0'),pair('13.7','11.0'),'.132','.130',pair('28.4','5.0'),pair('1.7','7.0'),pair('19.9','6.0')],
      ['DeepSeek-V4-Flash','A-MEM',pair('27.1','10.6'),pair('56.8','31.9','','second'),'.608','.610',pair('37.7','14.2'),pair('8.8','12.9'),pair('28.5','13.8')],
      ['DeepSeek-V4-Flash','HippoRAG',pair('27.0','11.1'),pair('55.9','30.6'),'.584','.574',pair('62.9','18.5'),pair('15.9','18.4'),pair('48.0','18.5')],
      ['DeepSeek-V4-Flash','Full context','—',pair('69.4','26.1'),'.592','.573','—','—','—'],
      ['DeepSeek-V4-Flash','SpeakerMem-R1†',pair('47.0','25.2','second','second'),pair('69.2','27.4','best'),best('.713'),second('.691'),pair('71.2','33.7','second','second'),pair('37.4','16.7'),pair('60.5','28.3','second','best')],
      ['DeepSeek-V4-Flash','SpeakerMem-R1',pair('47.9','26.5','best','best'),pair('64.9','32.7','second','best'),second('.710'),best('.693'),pair('72.0','33.8','best','best'),pair('40.0','15.7','best'),pair('61.9','28.1','best','second')],
    ]
  },
  luna: {
    caption: 'Main benchmark results with GPT-5.6-luna. The question-level judge remains GPT-4o-mini.',
    note: 'The cross-model block tests whether the design remains useful when the model used for memory and answer generation changes. Full context remains a SocialMemBench-only reference.',
    rows: [
      ['GPT-5.6-luna','BM25',pair('46.2','26.6','best','second'),pair('30.0','21.1'),'.412','.452',pair('69.6','33.0'),pair('27.8','24.4','second','best'),pair('56.3','30.3','','second')],
      ['GPT-5.6-luna','Embed',pair('35.3','18.4'),pair('36.4','22.7'),'.519','.538',pair('58.5','28.8'),pair('24.5','23.3'),pair('47.8','27.1')],
      ['GPT-5.6-luna','Mem0',pair('20.27','8.1'),pair('13.97','14.9'),'.252','.262',pair('35.0','18.7'),pair('1.4','9.6'),pair('24.38','15.8')],
      ['GPT-5.6-luna','A-MEM',pair('26.58','9.8'),pair('46.56','24.9','','second'),'.607','.601',pair('53.4','27.4'),pair('11.8','16.9'),pair('40.17','24.1')],
      ['GPT-5.6-luna','HippoRAG',pair('27.11','11.9'),pair('61.30','28.0','second','best'),second('.726'),best('.723'),pair('75.1','35.2','best','second'),pair('20.1','21.9'),pair('57.62','30.9','second','best')],
      ['GPT-5.6-luna','Full context','—',pair('71.3','29.6'),'.769','.750','—','—','—'],
      ['GPT-5.6-luna','SpeakerMem-R1',pair('42.7','26.8','second','best'),pair('64.4','24.1','best'),best('.731'),second('.715'),pair('71.3','35.6','second','best'),pair('35.8','16.4','best'),pair('60.0','29.5','best')],
    ]
  }
};

const resultColumns = ['Model','Method','GroupMem','SocialMem','MeanQ','MeanN','EverMem MC','EverMem OE','EverMem All'];
const evermemRows = [
  ['MemoBase','60.09','12.85','18.00','64.68','36.77','30.60','17.05','29.59','38.78','36.21'],
  ['Mem0','55.40','11.24','6.33','66.17','52.46','51.87','22.73','31.36','36.22','39.92'],
  ['Zep','73.71','8.03','13.00','67.16','47.54','43.66','26.70','35.50','44.39','41.67'],
  ['MemOS','71.36','18.88','15.67','69.90','51.99','45.15','28.98','32.54','48.47','44.63'],
  ['RippleMem','92.02','22.09',second('21.33'),'78.11',second('71.66'),'58.96','31.25','36.69',best('53.06'),'54.75'],
  ['EverOS',best('94.37'),best('28.11'),'20.33',second('86.07'),'68.62',best('84.70'),second('39.77'),best('42.60'),second('52.04'),second('60.08')],
  ['SpeakerMem-R1',second('93.43'),second('24.10'),best('35.00'),best('87.06'),best('75.64'),second('80.22'),best('50.57'),second('40.83'),'43.88',best('62.33')]
].map((cells, index) => ({ cells, className: index === 6 ? 'ours' : '' }));
const locomoRows = [
  ['Mem0','67.13','51.15','55.51',second('72.93'),'66.88'],
  ['A-MEM','39.79','18.85','49.91','54.05','48.38'],
  ['MemOS',second('81.09'),second('67.49'),best('75.18'),'55.90',second('75.80')],
  ['Zep','61.70','41.35','49.31',best('76.60'),'65.99'],
  ['LightRAG',best('86.68'),best('84.04'),'60.75','71.88',best('79.87')],
  ['SpeakerMem-R1','77.88','41.13',second('70.72'),'40.62','67.34']
].map((cells, index) => ({ cells, className: index === 5 ? 'ours' : '' }));

const resultTabs = {
  main: { columns: resultColumns, caption: '', note: '' },
  evermem: {
    columns: ['Method','Single','Multi','Temp','Const','Proact','Update','Style','Skill','Role','Weighted total'],
    caption: 'Public EverMemBench comparison under the GPT-4.1-mini / Gemini-3-Flash configuration. The table covers all 2,400 questions and nine published behavior labels.',
    note: 'SpeakerMem-R1 answers 1,496 / 2,400 questions correctly (62.33%), above the public EverOS result of about 60.08% and RippleMem at about 54.75%. Public totals are reconstructed from rounded category results; SpeakerMem-R1 uses question-level records.',
    rows: evermemRows
  },
  locomo: {
    columns: ['Method','Single-hop','Multi-hop','Temporal','Open-domain','ALL (Non-AD)'],
    caption: 'LoCoMo category accuracy (%). LightRAG is reproduced using its official implementation; results for other memory systems are primarily sourced from the cited public comparisons. The metric is GPT-4o-mini judged accuracy.',
    note: 'SpeakerMem-R1 is used here as a two-person long-conversation boundary test. The full five-category breakdown appears in the expandable supplement.',
    rows: locomoRows
  }
};

function setRowsClass(rows) {
  return rows.map((cells) => {
    const method = cells[1] || cells[0] || '';
    let className = '';
    if (/SpeakerMem-R1/.test(method)) className += 'ours ';
    if (/Full context/.test(method)) className += 'full-context ';
    if (method === 'SpeakerMem-R1' || method === 'SpeakerMem-R1†') className += 'section-start ';
    return { cells, className: className.trim() };
  });
}

let activeModel = 'deepseek';
let activeResult = 'main';
function renderResult() {
  document.querySelector('#main-model-controls').hidden = activeResult !== 'main';
  document.querySelector('#result-panel').setAttribute('aria-label', document.querySelector('.result-tab.active').textContent);
  document.querySelector('#result-table').scrollLeft = 0;
  const config = activeResult === 'main' ? mainSets[activeModel] : resultTabs[activeResult];
  const rows = activeResult === 'main' ? setRowsClass(config.rows) : config.rows;
  document.querySelector('#result-caption').textContent = config.caption;
  renderTable('#result-table', activeResult === 'main' ? resultColumns : config.columns, rows);
  document.querySelector('#result-note').textContent = activeResult === 'main' ? config.note : config.note;
}

document.querySelectorAll('.model-switch').forEach((button) => button.addEventListener('click', () => {
  activeModel = button.dataset.model;
  document.querySelectorAll('.model-switch').forEach((item) => { item.classList.toggle('active', item === button); item.setAttribute('aria-pressed', item === button ? 'true' : 'false'); });
  renderResult();
}));
document.querySelectorAll('.result-tab').forEach((button) => button.addEventListener('click', () => {
  activeResult = button.dataset.result;
  document.querySelectorAll('.result-tab').forEach((item) => { item.classList.toggle('active', item === button); item.setAttribute('aria-selected', item === button ? 'true' : 'false'); });
  renderResult();
}));
renderResult();

const writerRows = [
  { cells: ['SFT (epoch 10)','Qwen2.5-3B','175 / 305','57.38 ± 0.33%','−14.10 pp'] },
  { cells: ['Writer-R1 (30 steps)','Qwen2.5-3B','208 / 305',bold('68.20 ± 0.66%'),'−3.28 pp'], className: 'ours' },
  { cells: ['LLM writer','DeepSeek-V4-Flash','218 / 305','71.48 ± 0.66%','—'] }
];
renderTable('#writer-table', ['Writing strategy','Writer','Correct / total','Acc. (mean ± std.)','Gap to LLM writer'], writerRows);

renderTable('#ablation-table', ['Configuration','SocialMem Acc.','GroupMem Acc.','EverMem Acc.','Δ SM','Δ GM','Δ EM'], [
  ['S1-only','54.80','42.01','48.92','−14.40','−5.89','−12.98'],
  ['S2-only','42.58','33.15','38.33','−26.62','−14.75','−23.57'],
  ['S1 + S2 (per-speaker)','65.37','43.22','55.92','−3.83','−4.68','−5.98'],
  ['S1 + S2 (group)','63.72','41.07','54.88','−5.48','−6.83','−7.02'],
  ['SpeakerMem-R1†','69.2','47.0','60.5','0.0','−0.9','−1.4'],
  { cells: ['SpeakerMem-R1',bold('64.9'),bold('47.9'),bold('61.9'),'−4.3','0.0','0.0'], className: 'ours' }
]);
renderTable('#topk-table', ['S1 top-k','S2 k=2 (Acc / F1)','S2 k=4 (Acc / F1)','S2 k=6 (Acc / F1)'], [
  ['5','49.18 / 22.89','56.39 / 23.85','59.34 / 24.35'],
  ['10','59.34 / 23.56','64.26 / 25.00','65.57 / 24.75'],
  ['20','68.20 / 25.88',pair('73.11','26.53','best','best'),'71.15 / 25.78']
]);

const gmAcc = [
  ['BM25','89.9','46.9','41.8','41.4','23.4','15.1','44.6'],['Embed','87.8','40.8','25.8','17.3','20.6','17.0','34.5'],['Mem0','88.5','10.2','7.1','2.5','5.6','9.4','21.6'],['A-MEM','80.6','22.4','15.9','5.6','13.1','25.5','27.1'],['HippoRAG','77.7','22.4','18.7','5.6','11.2','25.5','27.0'],['SpeakerMem-R1†','71.2','44.9','40.7','49.4','33.6','36.8',bold('47.0')],{cells:['SpeakerMem-R1','73.4','51.0','43.4','50.6','33.6','31.1',bold('47.9')],className:'ours'}
];
const smAcc = [
  ['BM25','25.3','24.7','4.5','66.2','38.0','25.9','29.8','20.6','10.0','28.6'],['Embed','39.4','28.4','13.6','62.5','53.3','46.3','40.9','25.6','40.0','38.1'],['Mem0','20.9','14.8','4.5','11.3','3.3','25.9','9.4','12.2','10.0','13.7'],['A-MEM','60.6','76.5','9.1','80.0','65.2','51.8','53.0','45.0','50.0','56.8'],['HippoRAG','60.6','71.6','4.5','68.8','63.0','48.1','53.0','47.7','60.0','55.9'],['Full context','74.3','51.9','18.2','73.8','77.2','53.7','77.3','68.3','70.0','69.4'],['SpeakerMem-R1†','73.9','69.1','9.1','90.0','78.3','46.3','76.2','59.9','70.0','69.2'],{cells:['SpeakerMem-R1','69.1','70.4','4.5','80.0','79.3','53.7','65.7','56.1','70.0',bold('64.9')],className:'ours'}
];
const gmF1 = [
  ['BM25','25.6','33.5','25.8','36.4','24.2','2.0','25.0'],['Embed','24.2','27.0','17.1','14.2','22.4','4.4','17.4'],['Mem0','26.0','7.0','5.0','0.0','13.0','1.0','9.0'],['A-MEM','15.0','15.8','8.3','5.9','19.7','4.8','10.6'],['HippoRAG','14.6','12.4','11.3','6.2','19.7','4.5','11.1'],['SpeakerMem-R1†','14.5','26.6','22.3','49.7','26.5','5.1','25.2'],{cells:['SpeakerMem-R1','15.1','25.6','26.3','50.9','26.7','4.4',bold('26.5')],className:'ours'}
];
const smF1 = [
  ['BM25','13.8','10.3','12.1','38.1','18.7','14.5','15.7','12.2','8.2','15.7'],['Embed','15.2','12.5','10.0','39.8','20.8','18.4','18.6','15.8','7.8','18.1'],['Mem0','12.0','7.0','16.0','7.0','10.0','12.0','11.0','10.0','10.0','11.0'],['A-MEM','22.3','62.4','24.1','67.3','29.2','51.9','23.4','24.7','20.9','31.9'],['HippoRAG','21.2','60.1','23.4','55.7','30.6','49.0','23.1','25.0','19.0','30.6'],['Full context','22.6','26.8','8.5','47.3','30.0','31.9','24.1','23.3','16.9','26.1'],['SpeakerMem-R1†','27.9','5.9','26.5','35.4','34.7','4.6','30.8','30.9','33.9','27.4'],{cells:['SpeakerMem-R1','22.8','56.5','25.0','70.9','30.9','52.4','24.9','26.0','22.4',bold('32.7')],className:'ours'}
];
renderTable('#gm-acc-table',['Method','Abst.','Implicit','Multi-hop','Temporal','K.-update','Term-amb.','Total'],gmAcc);
renderTable('#sm-acc-table',['Method','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Total'],smAcc);
renderTable('#gm-f1-table',['Method','Abst.','Implicit','Multi-hop','Temporal','K.-update','Term-amb.','Total'],gmF1);
renderTable('#sm-f1-table',['Method','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Total'],smF1);

const gmLuna = [
  {cells:['SpeakerMem-R1','57.6','49.0','42.3','42.0','26.2','38.7','42.7'],className:'ours'},['BM25','71.9','49.0','43.4','56.8','27.1','18.9','46.2'],['Embed','70.5','44.9','29.7','25.3','19.6','25.5','35.3'],['Mem0','82.0','20.4','6.6','0.6','0.9','12.3','20.27'],['A-MEM','65.5','30.6','21.4','1.2','13.1','34.9','26.58'],['HippoRAG','64.7','28.6','22.5','4.9','13.1','33.0','27.11']
];
const smLuna = [
  {cells:['SpeakerMem-R1','68.7','72.8','4.5','83.8','80.4','53.7','58.6','56.9','80.0','64.4'],className:'ours'},['BM25','25.7','33.3','4.5','75.0','35.9','37.0','30.9','16.8','40.0','30.0'],['Embed','35.7','39.5','4.5','66.2','52.2','53.7','34.8','21.8','30.0','36.4'],['Mem0','20.1','21.0','9.1','17.5','0.0','31.5','7.2','11.5','10.0','13.97'],['A-MEM','54.6','63.0','4.5','70.0','54.3','38.9','43.1','32.1','30.0','46.56'],['HippoRAG','61.4','77.8','4.5','77.5','69.6','61.1','62.4','51.9','70.0','61.30'],['Full context','71.1','79.0','4.5','82.5','71.7','66.7','72.9','69.8','100.0','71.3']
];
renderTable('#gm-luna-table',['Method','Abst.','Implicit','Multi-hop','Temporal','K.-update','Term-amb.','Total'],gmLuna);
renderTable('#sm-luna-table',['Method','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Total'],smLuna);

const tokenCols = ['System','Calls / units','Ingest in.','Ingest out.','QA input','Answer out.','Total','Avg. / question'];
const tokenOverview = [
  ['SpeakerMem-R1 + ASK','7,212','25,546,058','721,200','13,865,187','430,530','40,562,975','9,713.36'],['BM25','0','0','0','5,298,971','85,011','5,383,982','1,289.27'],['Embed','0 (local)','0','0','5,096,215','96,764','5,192,979','1,243.53'],['Mem0 · chunk5','35,676','29,970,000','29,970,000','2,316,690','91,476','62,348,166','14,930.12'],['A-MEM · chunk5','35,676','39,929,508','20,100,569','12,482,895','159,830','72,672,802','17,402.49'],['HippoRAG · chunk5','35,676','30,400,001','30,400,001','11,506,209','155,857','72,462,068','17,352.03']
];
const tokenSocial = [
  ['SpeakerMem-R1 + ASK','348','845,605','34,800','846,404','75,247','1,802,056','1,747.87'],['BM25','0','0','0','1,308,247','20,988','1,329,235','1,289.27'],['Embed','0','0','0','1,258,189','23,890','1,282,079','1,243.53'],['Mem0 · chunk5','1,471','1,236,000','1,236,000','306,976','31,561','2,810,537','2,726.03'],['A-MEM · chunk5','1,471','1,646,400','828,800','1,328,327','73,915','3,877,442','3,760.86'],['HippoRAG · chunk5','1,471','1,253,473','1,253,473','1,032,295','71,444','3,610,685','3,502.12']
];
const tokenGroup = [
  ['SpeakerMem-R1 + ASK','4,815','17,445,456','481,500','2,152,506','29,344','20,108,806','26,991.69'],['BM25','0','0','0','945,338','15,166','960,504','1,289.27'],['Embed','0','0','0','909,167','17,263','926,430','1,243.53'],['Mem0 · chunk5','24,000','20,160,000','20,160,000','253,800','20,582','40,594,382','54,489.10'],['A-MEM · chunk5','24,000','26,861,726','13,522,230','2,444,228','33,277','42,861,461','57,532.16'],['HippoRAG · chunk5','24,000','20,450,953','20,450,953','2,448,714','32,569','43,383,189','58,232.47']
];
const tokenEver = [
  ['SpeakerMem-R1 + ASK','2,049','7,254,997','204,900','10,866,277','325,939','18,652,113','7,771.71'],['BM25','0','0','0','3,045,386','48,857','3,094,243','1,289.27'],['Embed','0','0','0','2,928,859','55,611','2,984,470','1,243.53'],['Mem0 · chunk5','10,205','8,574,000','8,574,000','1,755,914','39,333','18,943,247','7,893.02'],['A-MEM · chunk5','10,205','11,421,382','5,749,539','8,710,340','52,638','25,933,899','10,805.79'],['HippoRAG · chunk5','10,205','8,695,575','8,695,575','8,025,200','51,844','25,468,194','10,611.75']
];
renderTable('#token-overview-table',tokenCols,tokenOverview); renderTable('#token-social-table',tokenCols,tokenSocial); renderTable('#token-group-table',tokenCols,tokenGroup); renderTable('#token-ever-table',tokenCols,tokenEver);

renderTable('#locomo-breakdown-table',['Category','Correct / total','Accuracy (%)'],[
  ['Single-hop','655 / 841','77.88'],['Adversarial / AD','370 / 446','82.96'],['Temporal','227 / 321','70.72'],['Multi-hop','116 / 282','41.13'],['Open-domain','39 / 96','40.62'],{cells:['All questions','1,407 / 1,986',bold('70.85')],className:'ours'},['Excluding AD','1,037 / 1,540','67.34']
]);

const demos = {
  attribution: {
    description: 'A report about Bob stays attached to Alice as its source, while the fact itself is stored under Bob.',
    pipeline: [['Project', 'Target Bob and the PERSON scope.'], ['Select', 'Recall the matching verbatim messages and Bob\'s derived records.'], ['Resolve', 'Keep Alice as source and Bob as owner.'], ['Compose', 'Answer with the time and its attribution.']],
    messages: [
      {id:'e01', speaker:'Alice', time:'09:10', text:"Bob's train leaves at eight tomorrow."},
      {id:'e02', speaker:'Bob', time:'09:12', text:'Yes, I need to leave at eight.'},
      {id:'e03', speaker:'Chen', time:'09:15', text:"I'll pick Bob up at the station."}
    ],
    records: [
      {track:'raw', entry_id:'e01', owner:'Alice', source:'Alice', layer:'per_speaker_episodic', utype:'utterance', content:"Bob's train leaves at eight tomorrow.", from_ids:[]},
      {track:'raw', entry_id:'e02', owner:'Bob', source:'Bob', layer:'per_speaker_episodic', utype:'utterance', content:'Yes, I need to leave at eight.', from_ids:[]},
      {track:'raw', entry_id:'e03', owner:'Chen', source:'Chen', layer:'per_speaker_episodic', utype:'utterance', content:"I'll pick Bob up at the station.", from_ids:[]},
      {track:'derived', entry_id:'c01', owner:'Bob', source:'Alice', layer:'per_speaker_profile', utype:'observation', content:"Bob's train leaves at eight tomorrow.", from_ids:['e01']},
      {track:'derived', entry_id:'c02', owner:'Bob', source:'Bob', layer:'per_speaker_core', utype:'fact', content:'Bob needs to leave at eight tomorrow.', from_ids:['e02']}
    ],
    query: 'According to Alice, what time does Bob\'s train leave?',
    answer: "Alice reports that Bob's train leaves at eight tomorrow. The source is Alice; the owner of the information is Bob."
  },
  decision: {
    description: 'A personal preference remains on Mina’s PERSON row; a later group decision is stored on the GROUP row.',
    pipeline: [['Project', 'Track the dinner issue for Mina and the GROUP.'], ['Select', 'Retrieve Mina\'s preference and the later decision.'], ['Resolve', 'Separate a personal stance from a group decision.'], ['Compose', 'Return both facts with their scopes.']],
    messages: [
      {id:'e01', speaker:'Mina', time:'14:02', text:'I would rather eat at home.'},
      {id:'e02', speaker:'Jae', time:'14:05', text:'The new noodle shop is open.'},
      {id:'e03', speaker:'Mina', time:'14:07', text:'Okay, we agreed: we will eat at the noodle shop.'}
    ],
    records: [
      {track:'raw', entry_id:'e01', owner:'Mina', source:'Mina', layer:'per_speaker_episodic', utype:'utterance', content:'I would rather eat at home.', from_ids:[]},
      {track:'raw', entry_id:'e02', owner:'Jae', source:'Jae', layer:'per_speaker_episodic', utype:'utterance', content:'The new noodle shop is open.', from_ids:[]},
      {track:'raw', entry_id:'e03', owner:'Mina', source:'Mina', layer:'per_speaker_episodic', utype:'utterance', content:'Okay, we agreed: we will eat at the noodle shop.', from_ids:[]},
      {track:'derived', entry_id:'c01', owner:'Mina', source:'Mina', layer:'per_speaker_core', utype:'stance', content:'Mina would rather eat at home.', from_ids:['e01']},
      {track:'derived', entry_id:'c02', owner:'GROUP', source:'Mina', layer:'group_interaction', utype:'decision', content:'The group will eat at the new noodle shop.', from_ids:['e03']}
    ],
    query: 'What did the group decide, and what was Mina\'s personal preference?',
    answer: 'The group decided to eat at the new noodle shop. Mina personally preferred eating at home; the decision therefore belongs to GROUP, not to Mina alone.'
  },
  update: {
    description: 'An UPDATE appends a new state to the same entry chain. The earlier state remains available for a historical query.',
    pipeline: [['Project', 'Target Alex\'s plant-watering plan and the requested time mode.'], ['Select', 'Retrieve the current head and the linked earlier state.'], ['Resolve', 'Use Saturday for now and retain Friday for history.'], ['Compose', 'Answer the current and earlier plans together.']],
    messages: [
      {id:'e01', speaker:'Alex', time:'Mon 18:10', text:'I can water the plants on Friday.'},
      {id:'e02', speaker:'Alex', time:'Wed 08:20', text:'Friday is busy; I can do it on Saturday.'},
      {id:'e03', speaker:'Priya', time:'Wed 08:25', text:'Saturday works for me.'}
    ],
    records: [
      {track:'raw', entry_id:'e01', owner:'Alex', source:'Alex', layer:'per_speaker_episodic', utype:'utterance', content:'I can water the plants on Friday.', from_ids:[]},
      {track:'raw', entry_id:'e02', owner:'Alex', source:'Alex', layer:'per_speaker_episodic', utype:'utterance', content:'Friday is busy; I can do it on Saturday.', from_ids:[]},
      {track:'raw', entry_id:'e03', owner:'Priya', source:'Priya', layer:'per_speaker_episodic', utype:'utterance', content:'Saturday works for me.', from_ids:[]},
      {track:'derived', action:'ADD', entry_id:'c01', owner:'Alex', source:'Alex', layer:'per_speaker_core', utype:'stance', content:'Alex can water the plants on Friday.', from_ids:['e01'], superseded_by:'c02'},
      {track:'derived', action:'UPDATE', entry_id:'c02', owner:'Alex', source:'Alex', layer:'per_speaker_core', utype:'stance', content:'Alex can water the plants on Saturday.', from_ids:['e02'], links:['c01']}
    ],
    query: 'What day can Alex water the plants now, and what was the earlier plan?',
    answer: 'The current state is Saturday. The earlier Friday state is still retained in the chain, so the system can answer both current and historical questions.'
  },
  coverage: {
    description: 'The roster is explicit: the system can report who has made a choice and who has not, without inventing a group consensus.',
    pipeline: [['Project', 'Expand the roster so every member has a row.'], ['Select', 'Retrieve one relevant record per member.'], ['Resolve', 'Keep Owen\'s explicit non-decision as an observation.'], ['Compose', 'List choices and state what remains unknown.']],
    messages: [
      {id:'e01', speaker:'Noah', time:'10:00', text:'I can bring sandwiches.'},
      {id:'e02', speaker:'Sara', time:'10:03', text:'I will bring fruit.'},
      {id:'e03', speaker:'Owen', time:'10:05', text:"I haven't decided what to bring."},
      {id:'e04', speaker:'Lin', time:'10:08', text:'I can bring drinks.'}
    ],
    records: [
      {track:'raw', entry_id:'e01', owner:'Noah', source:'Noah', layer:'per_speaker_episodic', utype:'utterance', content:'I can bring sandwiches.', from_ids:[]},
      {track:'raw', entry_id:'e02', owner:'Sara', source:'Sara', layer:'per_speaker_episodic', utype:'utterance', content:'I will bring fruit.', from_ids:[]},
      {track:'raw', entry_id:'e03', owner:'Owen', source:'Owen', layer:'per_speaker_episodic', utype:'utterance', content:"I haven't decided what to bring.", from_ids:[]},
      {track:'raw', entry_id:'e04', owner:'Lin', source:'Lin', layer:'per_speaker_episodic', utype:'utterance', content:'I can bring drinks.', from_ids:[]},
      {track:'derived', entry_id:'c01', owner:'Owen', source:'Owen', layer:'per_speaker_profile', utype:'observation', content:'Owen has not decided what to bring.', from_ids:['e03']}
    ],
    query: 'Who has chosen what to bring, and who has not decided?',
    answer: 'Noah will bring sandwiches, Sara will bring fruit, and Lin will bring drinks. Owen has not decided; the record does not turn that into a guessed preference.'
  },
  norm: {
    description: 'A repeated group practice is stored as a GROUP insight, while the messages that support it remain verbatim.',
    pipeline: [['Project', 'Target the GROUP and the question about a shared practice.'], ['Select', 'Retrieve the relevant planning messages and the group insight.'], ['Resolve', 'Keep the norm at GROUP scope instead of assigning it to one speaker.'], ['Compose', 'State the practice and point back to its supporting messages.']],
    messages: [
      {id:'e01', speaker:'Mia', time:'17:20', text:"Let's share weekend plans here before we book anything."},
      {id:'e02', speaker:'Leo', time:'17:24', text:'I posted the dinner plan here before booking the table.'},
      {id:'e03', speaker:'Sara', time:'17:26', text:'That makes it easy for everyone to see the plan.'}
    ],
    records: [
      {track:'raw', entry_id:'e01', owner:'Mia', source:'Mia', layer:'per_speaker_episodic', utype:'utterance', content:"Let's share weekend plans here before we book anything.", from_ids:[]},
      {track:'raw', entry_id:'e02', owner:'Leo', source:'Leo', layer:'per_speaker_episodic', utype:'utterance', content:'I posted the dinner plan here before booking the table.', from_ids:[]},
      {track:'raw', entry_id:'e03', owner:'Sara', source:'Sara', layer:'per_speaker_episodic', utype:'utterance', content:'That makes it easy for everyone to see the plan.', from_ids:[]},
      {track:'derived', entry_id:'c01', owner:'GROUP', source:'Mia', layer:'group_insight', utype:'observation', content:'The group shares weekend plans in the chat before booking.', from_ids:['e01','e02','e03']}
    ],
    query: "What is the group's norm for weekend plans?",
    answer: "The group shares weekend plans in the chat before booking. This is a GROUP insight supported by the messages, not Mia's private preference."
  }
};
function renderRecord(record) {
  const fields = [
    ['owner', record.owner], ['source', record.source], ['layer', record.layer],
    ['utype', record.utype], ['content', record.content], ['from_ids', `[${record.from_ids.join(', ')}]`]
  ];
  if (record.superseded_by) fields.push(['superseded_by', record.superseded_by]);
  if (record.links) fields.push(['links', `[${record.links.join(', ')}]`]);
  const action = record.action ? `<span class="record-action ${record.action.toLowerCase()}">${record.action}</span>` : '';
  return `<article class="memory-record ${record.track}"><div class="record-top"><span class="memory-tag">${esc(record.layer)}</span>${action}<code>${esc(record.entry_id)}</code></div><dl>${fields.map(([key,value]) => `<div><dt>${esc(key)}</dt><dd>${esc(value)}</dd></div>`).join('')}</dl></article>`;
}
function renderDemo(key) {
  document.querySelector('.demo-trace-scroll').scrollTop = 0;
  const demo = demos[key];
  document.querySelector('#demo-messages').innerHTML = demo.messages.map((message) => `<div class="message"><b>${esc(message.speaker)} · ${esc(message.time)} <span class="message-id">${esc(message.id)}</span></b><span>${esc(message.text)}</span></div>`).join('');
  const raw = demo.records.filter((record) => record.track === 'raw');
  const derived = demo.records.filter((record) => record.track === 'derived');
  const pipeline = demo.pipeline.map(([stage, detail]) => `<div class="pipeline-step"><span>${esc(stage)}</span><p>${esc(detail)}</p></div>`).join('');
  document.querySelector('#demo-memory').innerHTML = `<p class="demo-memory-note">${esc(demo.description)}</p><div class="memory-track-title"><span>System 1</span><code>per_speaker_episodic · verbatim</code></div>${raw.map(renderRecord).join('')}<div class="memory-track-title derived-title"><span>System 2</span><code>four derived layers</code></div>${derived.map(renderRecord).join('')}<div class="demo-pipeline"><div class="memory-track-title pipeline-title"><span>Query path</span><code>Project → Select → Resolve → Compose</code></div>${pipeline}</div>`;
  document.querySelector('#demo-query-text').textContent = demo.query;
  document.querySelector('#demo-answer-text').textContent = demo.answer;
}
document.querySelectorAll('.demo-tab').forEach((button) => button.addEventListener('click', () => {
  document.querySelectorAll('.demo-tab').forEach((item) => { item.classList.toggle('active', item === button); item.setAttribute('aria-selected', item === button ? 'true' : 'false'); });
  renderDemo(button.dataset.demo);
}));
renderDemo('attribution');

const observer = typeof IntersectionObserver === 'function' ? new IntersectionObserver((entries) => entries.forEach((entry) => { if (entry.isIntersecting) entry.target.classList.add('visible'); }), { threshold: .08 }) : null;
document.querySelectorAll('.reveal').forEach((item) => { if (observer) { item.classList.add('reveal-pending'); observer.observe(item); } });
const header = document.querySelector('.site-header');
window.addEventListener('scroll', () => header.classList.toggle('scrolled', window.scrollY > 12), { passive: true });
const navItems = [...document.querySelectorAll('.nav-links a')];
const sections = [...document.querySelectorAll('main section[id]')].filter((section) => navItems.some((link) => link.getAttribute('href') === `#${section.id}`));
const navObserver = typeof IntersectionObserver === 'function' ? new IntersectionObserver((entries) => entries.forEach((entry) => { if (entry.isIntersecting) navItems.forEach((link) => link.classList.toggle('active', link.getAttribute('href') === `#${entry.target.id}`)); }), { rootMargin: '-35% 0px -55% 0px', threshold: 0 }) : null;
if (navObserver) sections.forEach((section) => navObserver.observe(section));

const modal = document.querySelector('#image-modal');
const modalImage = document.querySelector('#modal-image');
const modalCaption = document.querySelector('#modal-caption');
document.querySelectorAll('.image-trigger').forEach((trigger) => trigger.addEventListener('click', () => { modalImage.src = trigger.dataset.image; modalImage.alt = trigger.querySelector('img')?.alt || 'Research figure'; modalCaption.textContent = modalImage.alt; modal.showModal(); }));
document.querySelector('#modal-close').addEventListener('click', () => modal.close());
modal.addEventListener('click', (event) => { if (event.target === modal) modal.close(); });
document.querySelector('#bibtex-button')?.addEventListener('click', () => { const panel = document.querySelector('#bibtex-panel'); panel.hidden = !panel.hidden; });
document.querySelector('#copy-bibtex')?.addEventListener('click', async () => { const text = document.querySelector('#bibtex-panel pre').textContent; try { await navigator.clipboard?.writeText(text); } catch (_) {} const button = document.querySelector('#copy-bibtex'); button.textContent = 'Copied'; setTimeout(() => { button.textContent = 'Copy BibTeX'; }, 1400); });

// Arrow-key navigation for the result and example tab lists.
document.querySelectorAll('[role="tablist"]').forEach((list) => {
  const tabs = [...list.querySelectorAll('[role="tab"]')];
  const sync = () => tabs.forEach(tab => { tab.tabIndex = tab.getAttribute('aria-selected') === 'true' ? 0 : -1; });
  tabs.forEach(tab => tab.addEventListener('click', sync));
  list.addEventListener('keydown', event => {
    if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
    event.preventDefault();
    const index = tabs.indexOf(document.activeElement);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    tabs[next].click(); tabs[next].focus();
  });
  sync();
});
