//   navbar function 
$(document).ready(function(){

    $('.fa-bars').click(function(){
        $(this).toggleClass('fa-times');
        $('.navbar').toggleClass('nav-toggle');
    });

    $(window).on('scroll load',function(){
        $('.fa-bars').removeClass('fa-times');
        $('.navbar').removeClass('nav-toggle');

        if($(Window).scrollTop()  >  30){
            $('header').addClass('header-active');
        }else{
            $('header').removeClass('header-active');
        }
    });

    
});


// Sample code for initializing visualizations
document.addEventListener('DOMContentLoaded', function() {
    // Initialize prevalence graph
    initPrevalenceChart();
    
    // Load research studies
    loadResearchStudies();
    
    // Initialize 3D brain model
    initBrainModel();
});

function initPrevalenceChart() {
    // Use Chart.js or D3.js to create the chart
    // This would be replaced with actual chart initialization code
    console.log("Initializing prevalence chart...");
}

function loadResearchStudies() {
    // Fetch research data from API
    fetch('/api/research')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById('research-results');
            container.innerHTML = data.map(study => `
                <div class="research-card">
                    <div class="study-image">
                        <img src="${study.image || 'default-research.jpg'}" alt="${study.title}">
                    </div>
                    <div class="study-content">
                        <h3>${study.title}</h3>
                        <p class="study-meta">${study.authors} • ${study.date}</p>
                        <p class="study-excerpt">${study.abstract.substring(0, 150)}...</p>
                        <a href="/study/${study.id}" class="read-more">Read Study</a>
                    </div>
                </div>
            `).join('');
        });
}
document.addEventListener('DOMContentLoaded', function () {
  // Navbar scroll effect
  window.addEventListener('scroll', function () {
    const navbar = document.getElementById('navbar');
    navbar.classList.toggle('sticky', window.scrollY > 0);
  });

  // Chart.js setup
  const ctx = document.getElementById('prevalenceChart').getContext('2d');
  const prevalenceChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['60-69', '70-79', '80-89', '90+'],
      datasets: [{
        label: 'Prevalence (%)',
        data: [4.2, 13.1, 24.6, 35.8], // Replace with real data
        backgroundColor: '#4a90e2',
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        title: { display: true, text: 'Alzheimer’s Prevalence by Age Group' }
      },
      scales: {
        y: { beginAtZero: true, max: 40 }
      }
    }
  });

  // Prediction Form Logic (Mock Result)
  const form = document.getElementById('riskForm');
  form.addEventListener('submit', function (e) {
    e.preventDefault();
    const age = parseInt(document.getElementById('age').value);
    const gender = document.getElementById('gender').value;
    const cogScore = parseFloat(document.getElementById('cogScore').value);

    // Mock prediction logic (replace with real ML model later)
    let risk = 'Low';
    if (age > 70 && cogScore < 24) {
      risk = 'High';
    } else if (age > 60 && cogScore < 27) {
      risk = 'Moderate';
    }

    document.getElementById('predictionResult').innerHTML = `
      <p><strong>Estimated Risk:</strong> ${risk}</p>
    `;
  });
});
