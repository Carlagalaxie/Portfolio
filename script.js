document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".accordion-title").forEach(title => {
    title.addEventListener("click", () => {
      const parent = title.closest(".accordion");
      parent.classList.toggle("open");
    });
  });
});

// Script pour le bouton de changement de langue

// script.js

// Dictionnaire de traduction
const translations = {
  "pageTitle": "My Portfolio",
  "headerTitle": "My Portfolio",
  "navBio": "Biography",
  "navCV": "CV",
  "navStage": "M1 Internship",
  "navProjets": "Personal Projects",
  "navProjetsCours": "Course Projects",
  "navSeance1": "Session 1",
  "navSeance2": "Session 2",
  "navSeance3": "Session 3",
  "navSeance4": "Session 4",
  "navSeance5": "Session 5",
  "navSeance6": "Session 6",
  "langBtn": "FR",
  "bioTitle": "Biography",
  "bioText": "I am a second-year Master's in Modeling and Numerical Analysis at the University of Montpellier. Passionate about applied mathematics and numerical simulation, I am particularly interested in solving complex problems using numerical methods and mathematical models. My education has allowed me to develop skills in scientific programming and modeling of physical systems. Curious and rigorous, I seek to apply my knowledge to innovative projects in modeling and simulation.",
  "cvTitle": "CV",
  "btnCV": "My CV",
  "btnEmail": "Email",
  "btnLinkedIn": "LinkedIn",
  "skillsTech": "Technical Skills (Modeling & Scientific Computing)",
  "skillSim": "Numerical simulation: Experience in simulating compressible gas dynamics.",
  "skillNum": "Numerical methods: Mastery of finite elements, finite volumes, and finite differences.",
  "skillSolver": "Solver development: Implementation and comparison of Riemann solvers for numerical flux analysis.",
  "skillLang": "Programming languages: Proficient in Python (Jupyter, VS Code, NumPy, Matplotlib), C++, MATLAB, R, and Java.",
  "skillTools": "Software & Tools: Experience with FreeFem++, Office Suite, and basic web development (HTML/CSS).",
  "skillsPers": "Personal Skills",
  "skillPers1": "Rigor, autonomy, organization, and adaptability",
  "skillPers2": "Experience in customer service and teamwork (seasonal work).",
  "skillPersLang": "Languages: English (B1 level)",
  "stageTitle": "Master 1 Internship",
  "stageDescTitle": "Internship Description:",
  "stageDesc1": "Internship completed at the Institute of Mathematics of Montpellier under the supervision of Mr. VILAR. This Master 1 internship in applied mathematics and modeling focused on numerical simulation of compressible gas flows governed by the Euler equations. These hyperbolic conservation equations are essential in fields such as aeronautics and aerospace.",
  "stageDesc2": "The main goal of the internship was to implement and compare several approximate Riemann solvers: Rusanov, HLL, HLLC, Roe, to effectively resolve physical discontinuities (shocks, rarefactions, contacts). The analysis focused on the robustness, accuracy, and positivity of these methods, followed by validation on classical test cases (Sod tube, contact discontinuity).",
  "stageSkillsTitle": "Skills utilized:",
  "stageSkill1": "Mathematical modeling (hyperbolic PDEs)",
  "stageSkill2": "Numerical methods (finite volumes, Riemann solvers)",
  "stageSkill3": "Scientific programming (Python)",
  "stageSkill4": "Critical analysis of numerical results",
  "stageReportTitle": "Written Report: Compressible Gas Dynamics",
  "stageReportBtn": "Numerical Modeling of Compressible Gas Dynamics",
  "projectsTitle": "Personal Projects",
  "project1Title": "Personal Project / Other Course",
  "project1Desc": "Description.",
  "project2Title": "Another Project",
  "project2Desc": "Description of a second project.",
  "courseProjectsTitle": "Course Projects: A Posteriori Estimation",
  "seance1Title": "Session 1",
  "seance1Desc": "The purpose of session 1 is to solve an ODE using explicit Euler, compare with the exact solution, and analyze the error as a function of the time step.",
  "seance1Btn1": "Download Euler_ODE_Errors",
  "seance1Btn2": "2D",
  "seance1Btn3": "Session 1 Explanation",
  "seance2Title": "Session 2",
  "seance2Desc": "The adrs.py code numerically solves a 1D ADRS (advection–diffusion–reaction–source) equation using finite differences. It then compares the numerical solution with a manufactured exact solution. It also studies convergence with mesh refinement. Finally, it plots the final numerical solution, residual decay, and computes the numerical error.",
  "seance2Btn1": "Download adrsmodif",
  "seance2Btn2": "Download adrsmodif2",
  "seance2Btn3": "Download adrsmesh",
  "seance2Btn4": "Session 2 Explanation",
  "seance2Btn5": "Download integrale",
  "seance3Title": "Session 3",
  "seance3Desc": "...",
  "seance4Title": "Session 4",
  "seance4Desc": "Description of session 4.",
  "seance5Title": "Session 5",
  "seance5Desc": "Description of session 5.",
  "seance6Title": "Session 6",
  "seance6Desc": "Description of session 6.",
  "footerCopy": "© 2025 - My Portfolio",
  "footerEmail": "Email",
  "footerLinkedIn": "LinkedIn"
};

// Fonction pour changer la langue
function toggleLanguage() {
  const btn = document.getElementById('lang-toggle');
  const currentLang = btn.textContent;
  
  // On alterne FR/EN
  const isEnglish = currentLang === 'EN';
  btn.textContent = isEnglish ? 'FR' : 'EN';
  
  // Parcourir tous les éléments avec data-key
  document.querySelectorAll('[data-key]').forEach(el => {
    const key = el.getAttribute('data-key');
    if (isEnglish && translations[key]) {
      el.textContent = translations[key];
    } else if (!isEnglish && translations[key]) {
      // Revenir au français initial en utilisant l'attribut HTML original
      // Ici on peut stocker le texte original en data-fr au chargement si nécessaire
      el.textContent = el.getAttribute('data-fr') || el.textContent;
    }
  });
}

// Stocker le texte français original dans data-fr
document.querySelectorAll('[data-key]').forEach(el => {
  el.setAttribute('data-fr', el.textContent);
});

// Attacher l'événement au bouton
document.getElementById('lang-toggle').addEventListener('click', toggleLanguage);






// Drapeau changement de langue
function toggleLanguage() {
  const btn = document.getElementById('lang-toggle');
  const currentLang = btn.textContent.includes('EN') ? 'EN' : 'FR';
  const isEnglish = currentLang === 'EN';

  // Changer texte et icône
  btn.innerHTML = isEnglish
    ? '<img src="drapeau-fr.png" alt="FR" class="flag-icon"> FR'
    : '<img src="drapeau-gb.png" alt="GB" class="flag-icon"> EN';

  // Traduire le contenu
  document.querySelectorAll('[data-key]').forEach(el => {
    const key = el.getAttribute('data-key');
    if (!isEnglish && translations[key]) {
      el.textContent = translations[key]; // anglais
    } else {
      el.textContent = el.getAttribute('data-fr') || el.textContent; // français
    }
  });
}



//Luminosité
const themeToggle = document.getElementById('theme-toggle');
themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark');

    // Optionnel : stocker la préférence dans le localStorage
    if(document.body.classList.contains('dark')){
        localStorage.setItem('theme', 'dark');
    } else {
        localStorage.setItem('theme', 'light');
    }
});

// Charger la préférence au démarrage
window.addEventListener('load', () => {
    const savedTheme = localStorage.getItem('theme');
    if(savedTheme === 'dark') document.body.classList.add('dark');
});
