let questions = [];
let current = 0;
let answers = {};
let shuffledOptions = {};
let startTime = 0;
let timerInterval = null;

function shuffleArray(arr) {
  return arr
    .map(v => ({ v, sort: Math.random() }))
    .sort((a, b) => a.sort - b.sort)
    .map(({ v }) => v);
}

// ---------- LOAD QUESTIONS ----------

fetch("questions.json")
  .then(res => res.json())
  .then(data => {
    questions = shuffleArray(data); // перемешиваем
    // оставляем тест неактивным до старта
    document.getElementById("startQuiz").onclick = () => {
      const n = Number(document.getElementById("numQuestions").value) || 100;
      questions = questions.slice(0, n); // обрезаем до нужного числа
      document.getElementById("setup").style.display = "none";
      startQuiz();
    };
  });

function startQuiz() {
  current = 0;
  answers = {};
  shuffledOptions = {};
  startTime = Date.now();

  // старт таймера
  const timerEl = document.getElementById("timer");
  timerInterval = setInterval(() => {
    let seconds = Math.floor((Date.now() - startTime) / 1000);
    let m = String(Math.floor(seconds / 60)).padStart(2, "0");
    let s = String(seconds % 60).padStart(2, "0");
    timerEl.textContent = `⏱ ${m}:${s}`;
  }, 1000);

  document.getElementById("quiz").style.visibility = "visible";
  document.getElementById("next").parentElement.style.visibility = "visible";

  renderQuestion();
}

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
      q.options.map((text, index) => ({ text, index }))
    );
  }

  shuffledOptions[q.id].forEach((opt) => {
    let type = q.type === "multiple" ? "checkbox" : "radio";
    let checked =
      answers[q.id]?.includes(opt.index) ? "checked" : "";

    let label = document.createElement("label");
    label.innerHTML = `
      <input type="${type}" name="q${q.id}" value="${opt.index}" ${checked}>
      ${opt.text}
    `;
    opts.appendChild(label);
  });

  div.appendChild(opts);
  quiz.appendChild(div);
}

// ---------- SAVE + NEXT ----------
document.getElementById("next").onclick = () => {
  let q = questions[current];
  answers[q.id] = [...document.querySelectorAll(`input[name="q${q.id}"]:checked`)]
      .map(i => Number(i.value));

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

  // Скрываем quiz-блок и кнопку
  const quiz = document.getElementById("quiz");
  quiz.style.display = "none";
  const next = document.getElementById("next");
  next.parentElement.style.display = "none";
  document.getElementById("progress").style.display = "none";

  const results = document.getElementById("results");
  results.style.visibility = "visible";

  const question_results = document.getElementById("question_results");
  question_results.innerHTML = ""; // очистка

  let correctCount = 0;

  questions.forEach((q, index) => {
    let user = answers[q.id] || [];
    let ok =
      user.length === q.correct.length &&
      user.every(v => q.correct.includes(v));

    if (ok) {
      correctCount++;
    } else {
      let li = document.createElement("li");
      li.className = "wrong-card";

      li.innerHTML = `
        <div class="card-header">Вопрос ${index + 1}</div>
        <p>${q.question}</p>
        <div class="answer-block user-answer">
          <span>Ваш ответ:</span>
          <span>${user.map(i => q.options[i]).join(", ") || "—"}</span>
        </div>
        <div class="answer-block correct-answer">
          <span>Правильный:</span>
          <span>${q.correct.map(i => q.options[i]).join(", ")}</span>
        </div>
      `;

      question_results.appendChild(li);
    }
  });

  // Итог отдельно
  const summary = document.createElement("h3");
  summary.textContent = `Итого: ${correctCount} / ${questions.length}`;
  results.appendChild(summary);

  if (correctCount === questions.length) {
    document.getElementById("results_header").textContent = "Нет ошибок!"
  }
}
