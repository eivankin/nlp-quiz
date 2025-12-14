let questions = [];
let current = 0;
let answers = {};
let startTime = Date.now();
let timerInterval = null;
let shuffledOptions = {};

function shuffleArray(arr) {
  return arr
    .map(v => ({ v, sort: Math.random() }))
    .sort((a, b) => a.sort - b.sort)
    .map(({ v }) => v);
}

// ---------- TIMER ----------
const timerEl = document.getElementById("timer");

timerInterval = setInterval(() => {
  let seconds = Math.floor((Date.now() - startTime) / 1000);
  let m = String(Math.floor(seconds / 60)).padStart(2, "0");
  let s = String(seconds % 60).padStart(2, "0");
  timerEl.textContent = `⏱ ${m}:${s}`;
}, 1000);

// ---------- LOAD QUESTIONS ----------
fetch("questions.json")
  .then(res => res.json())
  .then(data => {
    questions = data;
    renderQuestion();
  });

// ---------- RENDER ----------
function renderQuestion() {
  const quiz = document.getElementById("quiz");
  const progress = document.getElementById("progress");
  quiz.innerHTML = "";

  let q = questions[current];
  progress.textContent = `Вопрос ${current + 1} из ${questions.length}`;

  let div = document.createElement("div");
  div.className = "question";

  div.innerHTML = `<h3>${q.question}</h3>`;

    let opts = document.createElement("div");
    opts.className = "options";

    if (!shuffledOptions[q.id]) {
        shuffledOptions[q.id] = shuffleArray(
            q.options.map((text, index) => ({text, index}))
        );
    }

  shuffledOptions[q.id].forEach((opt, i) => {
    let type = q.type === "multiple" ? "checkbox" : "radio";
    let checked =
      answers[q.id]?.includes(opt.index) ? "checked" : "";

    opts.innerHTML += `
      <label>
        <input type="${type}" name="q${q.id}" value="${opt.index}" ${checked}>
        ${opt.text}
      </label>
    `;
  });

  div.appendChild(opts);
  quiz.appendChild(div);
}

// ---------- SAVE + NEXT ----------
document.getElementById("next").onclick = () => {
  let q = questions[current];
  let selected = [...document.querySelectorAll(`input[name="q${q.id}"]:checked`)]
    .map(i => Number(i.value));

  answers[q.id] = selected;

  current++;

  if (current < questions.length) {
    renderQuestion();
  } else {
    finishQuiz();
  }
};

// ---------- FINISH ----------
function finishQuiz() {
  clearInterval(timerInterval);

  document.getElementById("quiz").innerHTML = "";
  document.getElementById("next").style.display = "none";
  document.getElementById("progress").style.display = "none";


  let results = document.getElementById("results");
  results.style.visibility = "visible";
  let question_results = document.getElementById("question_results");

  let correctCount = 0;

  questions.forEach(q => {
    let user = answers[q.id] || [];
    let ok =
      user.length === q.correct.length &&
      user.every(v => q.correct.includes(v));

    if (ok) correctCount++;

    if (!ok) {
        question_results.innerHTML += `
      <li value="${q.id}" class="${ok ? "correct" : "wrong"}">
        ${q.question}<br>
        Ваш ответ: ${user.map(i => q.options[i]).join(", ") || "—"}<br>
        Правильный: ${q.correct.map(i => q.options[i]).join(", ")}
      </li>
    `;
    }
  });

  results.innerHTML += `<h3>Итого: ${correctCount} / ${questions.length}</h3>`;
}
