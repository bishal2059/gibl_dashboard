import { marked } from 'marked';

// Global Application State
let appState = {
    currentUser: '',
    selectedBranch: null,
    selectedMonths: 0,
    currentTheme: 'light',
    activeTab: 'dashboard',
    activeSubtab: 'summary',
    charts: {},
    loadingComplete: false,
    chatExpanded: true
};


let appData = {
    summaryCards: '',
    analysisCards: '',
    keyInsights: '',
    followUpQuestions: ''
};

// Application Data
const applicationData = {
    kpis: {
        totalRevenue: 45200000,
        revenueGrowth: 8.3,
        activeBranches: 125,
        newBranches: 5,
        totalCustomers: 1200000,
        customerGrowth: 12.5,
        overallGrowthRate: 12.8,
        growthIncrease: 2.1
    }
};

// Utility Functions
const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-NP', {
        style: 'currency',
        currency: 'NPR',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount).replace('NPR', 'NPR ');
};

const formatNumber = (number) => {
    return new Intl.NumberFormat('en-US').format(number);
};

const showNotification = (message, type = 'success') => {
    const container = document.getElementById('notifications');
    if (!container) return;
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
            <span>${message}</span>
        </div>
    `;
    
    container.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 5000);
};

const generateId = () => {
    return Math.random().toString(36).substr(2, 9);
};

// Theme Management
const initializeTheme = () => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    appState.currentTheme = savedTheme;
    document.body.setAttribute('data-theme', savedTheme);
    updateThemeButton();
};

const toggleTheme = () => {
    appState.currentTheme = appState.currentTheme === 'light' ? 'dark' : 'light';
    document.body.setAttribute('data-theme', appState.currentTheme);
    localStorage.setItem('theme', appState.currentTheme);
    updateThemeButton();
    showNotification(`Switched to ${appState.currentTheme} theme`, 'success');
};

const updateThemeButton = () => {
    const themeToggle = document.getElementById('themeToggle');
    const themeOptions = document.querySelectorAll('.theme-option');
    
    if (themeToggle) {
        const icon = themeToggle.querySelector('i');
        icon.className = appState.currentTheme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
    }
    
    themeOptions.forEach(option => {
        option.classList.toggle('active', option.dataset.theme === appState.currentTheme);
    });
};

// Navigation Management
const switchTab = (tabId) => {
    appState.activeTab = tabId;
    
    // Update navigation
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabId);
    });
    
    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tabId}Tab`);
    });
    
    // Initialize tab-specific content
    if (tabId === 'insights') {
        switchSubtab('summary');
    }
};

const switchSubtab = (subtabId) => {
    appState.activeSubtab = subtabId;
    
    // Update subnav buttons
    document.querySelectorAll('.subnav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.subtab === subtabId);
    });
    
    // Update subtab content
    document.querySelectorAll('.subtab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${subtabId}Subtab`);
    });
};

// Loading Screen Management
const startLoadingSequence = () => {
    const loadingTexts = [
        "Initializing System...",
        "Loading Bank Data...",
        "Connecting to ML Models...",
        "Preparing Analytics...",
        "Finalizing Setup..."
    ];
    
    let currentIndex = 0;
    const loadingText = document.getElementById('loadingText');
    
    const updateLoadingText = () => {
        if (loadingText && currentIndex < loadingTexts.length) {
            loadingText.textContent = loadingTexts[currentIndex];
            currentIndex++;
            setTimeout(updateLoadingText, 1000);
        }
    };
    
    updateLoadingText();
    
    setTimeout(() => {
        appState.loadingComplete = true;
        showScreen('loginScreen');
    }, 4000);
};

const showScreen = (screenId) => {
    document.querySelectorAll('.screen').forEach(screen => {
        screen.classList.remove('active');
    });
    
    const targetScreen = document.getElementById(screenId);
    if (targetScreen) {
        targetScreen.classList.add('active');
    }
};

// Authentication
const handleLogin = (e) => {
    e.preventDefault();
    
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value;
    
    // Clear previous errors
    document.querySelectorAll('.error-message').forEach(error => {
        error.classList.remove('show');
    });
    
    // Validate inputs
    let isValid = true;
    
    if (!username) {
        showError('username', 'Username is required');
        isValid = false;
    }
    
    if (!password) {
        showError('password', 'Password is required');
        isValid = false;
    } else if (password.length < 4) {
        showError('password', 'Password must be at least 4 characters');
        isValid = false;
    }
    
    if (isValid) {
        appState.currentUser = username;
        const currentUserElement = document.getElementById('currentUser');
        if (currentUserElement) {
            currentUserElement.textContent = username;
        }
        showScreen('mainScreen');
        showNotification(`Welcome back, ${username}!`, 'success');
        const chatBox = document.getElementById('chatAssistant');
        chatBox.classList.remove('hidden');
    }
};

const showError = (fieldId, message) => {
    const errorElement = document.getElementById(fieldId + 'Error');
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.classList.add('show');
    }
};

const handleLogout = () => {
    appState.currentUser = '';
    appState.selectedBranch = null;
    appState.selectedMonths = 0;
    
    // Reset forms
    const forms = document.querySelectorAll('form');
    forms.forEach(form => form.reset());
    
    // Clear errors
    document.querySelectorAll('.error-message').forEach(error => {
        error.classList.remove('show');
    });

     chatBox = document.getElementById('chatAssistant');
     chatBox.classList.add('hidden');
    
    showScreen('loginScreen');
    showNotification('Logged out successfully', 'success');
};

// Prediction System
const handlePrediction = (e) => {
    e.preventDefault();
    
    const branchId = document.getElementById('branchSelect').value;
    const months = parseInt(document.getElementById('monthSelect').value);
    
    if (!branchId || !months) {
        showNotification('Please select both branch and prediction period', 'error');
        return;
    }
    
    // Show loading state
    const btnText = document.querySelector('.btn-text');
    const btnLoading = document.querySelector('.btn-loading');
    
    if (btnText && btnLoading) {
        btnText.classList.add('hidden');
        btnLoading.classList.remove('hidden');
    }
    
    appState.selectedMonths = months;
    //api call to fetch branch data
    fetch(`http://localhost:8000/api/predict/${branchId}/${months}`)
        .then(response => response.json())
        .then(data => {
            // console.log('Branch data fetched:',data);
            appState.selectedBranch = data
               // Simulate processing delay
    setTimeout(() => {
        generatePredictionResults();
        
        // Reset button state
        if (btnText && btnLoading) {
            btnText.classList.remove('hidden');
            btnLoading.classList.add('hidden');
        }
        
        // Show results
        const resultsSection = document.getElementById('predictionResults');
        if (resultsSection) {
            resultsSection.classList.remove('hidden');
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
        
        showNotification('Prediction generated successfully!', 'success');
    }, 2000);
        })
        .catch(error => {
            console.error('Error fetching branch data:', error);
            showNotification('Failed to fetch branch data', 'error');
        }).finally(() =>{
            fetchInsightsData(branchId);
        });
};

const generatePredictionResults = () => {
    if (!appState.selectedBranch) return;
    
    // Update results header
    const resultsTitle = document.getElementById('resultsTitle');
    const selectedBranchName = document.getElementById('selectedBranchName');
    
    if (resultsTitle) {
        resultsTitle.textContent = `Revenue Prediction - ${appState.selectedBranch.name}`;
    }
    if (selectedBranchName) {
        selectedBranchName.textContent = appState.selectedBranch.name;
    }
    
    // Generate prediction data
    const predictions = appState.selectedBranch.predictedRevenue;
    
    // Create chart
    createRevenueChart(appState.selectedBranch, predictions);
    
    // Populate table
    populatePredictionTable(predictions);
};


const createRevenueChart = (branchData, predictions) => {
    const canvas = document.getElementById('revenueChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Destroy existing chart
    if (appState.charts.revenue) {
        appState.charts.revenue.destroy();
    }
    
    const historicalLabels = branchData.historicalRevenue.map(item => item.month);
    const historicalData = branchData.historicalRevenue.map(item => item.revenue);
    const predictionLabels = predictions.map(item => item.month);
    const predictionData = predictions.map(item => item.revenue);
    
    const allLabels = [...historicalLabels, ...predictionLabels];
    const allHistoricalData = [...historicalData, ...new Array(predictions.length).fill(null)];
    const allPredictionData = [...new Array(historicalData.length).fill(null), ...predictionData];
    
    appState.charts.revenue = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allLabels,
            datasets: [
                {
                    label: 'Historical Revenue',
                    data: allHistoricalData,
                    borderColor: '#023570',
                    backgroundColor: 'rgba(2, 53, 112, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#023570',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 5
                },
                {
                    label: 'Predicted Revenue',
                    data: allPredictionData,
                    borderColor: '#c5161d',
                    backgroundColor: 'rgba(197, 22, 29, 0.1)',
                    borderWidth: 3,
                    borderDash: [10, 5],
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#c5161d',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Revenue Forecast: ${branchData.name}`,
                    font: {
                        size: 16,
                        weight: 'bold'
                    },
                    color: '#023570'
                },
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${formatCurrency(context.raw)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return formatCurrency(value);
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
};

const populatePredictionTable = (predictions) => {
    const tbody = document.getElementById('predictionTableBody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    predictions.forEach(prediction => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${prediction.month}</td>
            <td>${formatCurrency(prediction.revenue)}</td>
            <td><span class="status status--success">${prediction.confidence.toFixed(1)}%</span></td>
            <td class="${prediction.growth > 0 ? 'text-success' : 'text-error'}">${prediction.growth > 0 ? '+' : ''}${prediction.growth.toFixed(1)}%</td>
        `;
        tbody.appendChild(row);
    });
};

// Chat System
const initializeChatSystem = () => {
    const chatHeader = document.getElementById('chatHeader');
    const chatContent = document.getElementById('chatContent');
    const chatInput = document.getElementById('chatInput');
    const chatSend = document.getElementById('chatSend');
    const chatMinimize = document.getElementById('chatMinimize');
    
    if (!chatHeader || !chatContent || !chatInput || !chatSend) return;
    
    // Chat toggle
    if (chatMinimize) {
        chatMinimize.addEventListener('click', (e) => {
            e.stopPropagation();
            appState.chatExpanded = !appState.chatExpanded;
            chatContent.classList.toggle('collapsed', !appState.chatExpanded);
            
            const icon = chatMinimize.querySelector('i');
            icon.className = appState.chatExpanded ? 'fas fa-minus' : 'fas fa-plus';
        });
    }
    
    // Send message function
    const sendMessage = async () => {
        const message = chatInput.value.trim();
        if (!message) return;
        
        addChatMessage(message, 'user');
        chatInput.value = '';
        
        // get chat response
        const response = await generateChatResponse(message);
        // console.log('Chat response:', response);
        addChatMessage(response, 'bot');
    };
    
    // Event listeners
    chatSend.addEventListener('click', sendMessage);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key == 'Enter') {
            // console.log('Enter key pressed, sending message');
            sendMessage();
        }
    });
};

const addChatMessage = (message, sender) => {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}-message`;
    
    const currentTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas ${sender === 'bot' ? 'fa-robot' : 'fa-user'}"></i>
        </div>
        <div class="message-content">
            <p>${message}</p>
            <span class="message-time">${currentTime}</span>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
};
const generateChatResponse = async (userMessage) => {
    const qns = userMessage;

    try {
        const response = await fetch('http://localhost:8000/api/chatbot/insights', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: qns })
        });

        const data = await response.json();

        if (data.answer) {
            return marked(data.answer);
        } else {
            return "I'm sorry, I didn't understand that. Can you please rephrase?";
        }

    } catch (error) {
        console.error('Error fetching chatbot response:', error);
        return "I'm sorry, there was an error processing your request. Please try again later.";
    }
};

// insights data api call function
function fetchInsightsData(branch_Id) {
        //get sample data from api call
        fetch(`http://localhost:8000/api/current_insights/${branch_Id}`)
        .then(response => response.json())
        .then(sample_data => {
               // Populate insights data
            //    console.log('Sample insights data:', sample_data); // comment this line to hide sample data in console
            // populate insights data
            const insights = document.getElementById('insightsTab');
            insights.classList.remove('hidden');
            clearChat();
            populateInsightsData(sample_data);
            showNotification('Insights generated successfully', 'success');
        })
        .catch(error => {
            console.error('Error fetching insights data:', error);
            showNotification('Failed to fetch insights data', 'error');
        });
}

//populate insights data
const populateInsightsData = (insights) => {
        appData.summaryCards = insights["summary"];
        appData.analysisCards = insights["analysis"];
        appData.keyInsights = insights["insights_and_recommendations"];
        appData.followUpQuestions = insights["suggested_follow_up_questions"];
 

        summaryCards();
        loadAnalysisCards();
        loadInsightCards();
        loadFollowUpQuestions();
        

}

function summaryCards() {
    const summaryGrid = document.getElementById('summaryGrid');

    summaryGrid.innerHTML = Object.entries(appData.summaryCards).map(([key,value]) => {
        return `<div class="card">
            <div class="card__header">
                <h3> ${key}</h3>
            </div>
            <div class="card__body">
            <ul class="summary-list">
                ${generate_p(value)}
            </ul>
            </div>
        </div>
    `}).join('');
}

function loadAnalysisCards() {
    const analysisGrid = document.getElementById('analysisGrid');

    analysisGrid.innerHTML = Object.entries(appData.analysisCards).map(([key,value]) => {
        return `<div class="card">
            <div class="card__header">
                <h3> ${key}</h3>
            </div>
            <div class="card__body">
                ${generate_p(value)}
            </div>
        </div>
    `}).join('');
       
    }

function generate_p(text) {
    // generate p from each value in array
    if (Array.isArray(text)) {
        return text.map(item => `<li>${item}</li>`).join('');
    } else {
        return `<li>${text}</li>`;
    }
}

const options = ['low', 'medium', 'high'];

function loadInsightCards() {
    const insightsGrid = document.getElementById('insightsGrid');
    
    insightsGrid.innerHTML = Object.entries(appData.keyInsights).map(([key, value]) => `
        <div class="insight-card">
            <div class="insight-card__header">
                <div class="insight-card__severity insight-card__severity--${options[Math.floor(Math.random() * options.length)]}"></div>
                <h3 class="insight-card__title">${key}</h3>
            </div>
            <p class="insight-card__description">${value["points"] || value["insights"] || value["insight"]}</p>
            <div class="insight-card__recommendation">
                <strong>Recommendation:</strong> ${value["recommendations"]}
            </div>
        </div>
    `).join('');
}


function loadFollowUpQuestions() {
    const questionsList = document.getElementById('questionsList');
    
    questionsList.innerHTML = appData.followUpQuestions.map((question, index) => `
        <div class="question-item" data-question="${index}">
            <p class="question-item__text">${question}</p>
        </div>
    `).join('');
}




// Quick Actions
const handleQuickAction = (action) => {
    switch (action) {
        case 'predictions':
            switchTab('predictions');
            document.getElementById('branchSelect')?.focus();
            break;
        case 'insights':
            switchTab('insights');
            break;
            break;
        case 'reports':
            switchTab('reports');
            break;
        default:
            console.log('Unknown action:', action);
    }
};

// Export Functions
const exportReport = (format, reportType) => {
    showNotification(`Generating ${reportType} report in ${format.toUpperCase()} format...`, 'success');
    
    // Simulate export process
    setTimeout(() => {
        showNotification(`${reportType} report downloaded successfully!`, 'success');
    }, 2000);
};

// Settings Management
const handleThemeChange = (theme) => {
    if (theme !== appState.currentTheme) {
        toggleTheme();
    }
};

// Event Listeners Setup
const initializeEventListeners = () => {
    // Login form
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', handleLogin);
    }
    
    // Prediction form
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePrediction);
    }
    
    // Logout button
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }
    
    // Theme toggle
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
    
    // Navigation tabs
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            switchTab(e.target.dataset.tab);
        });
    });
    
    // Insights sub-navigation
    document.querySelectorAll('.subnav-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            switchSubtab(e.target.dataset.subtab);
        });
    });
    
    // Quick action buttons
    document.querySelectorAll('.action-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            handleQuickAction(e.currentTarget.dataset.action);
        });
    });
    
    // Export buttons
    document.querySelectorAll('.export-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const format = e.currentTarget.textContent.toLowerCase().includes('pdf') ? 'pdf' : 
                          e.currentTarget.textContent.toLowerCase().includes('excel') ? 'excel' : 
                          e.currentTarget.textContent.toLowerCase().includes('csv') ? 'csv' : 'powerpoint';
            exportReport(format, 'General');
        });
    });
    
    // Report generation buttons
    document.querySelectorAll('.report-card .btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const reportType = e.target.closest('.report-card').querySelector('h5').textContent;
            exportReport('pdf', reportType);
        });
    });
    
    // Theme selection in settings
    document.querySelectorAll('.theme-option').forEach(option => {
        option.addEventListener('click', (e) => {
            handleThemeChange(e.target.dataset.theme);
        });
    });
};

// Application Initialization
const initializeApplication = () => {
    console.log('Initializing Global IME Bank Enhanced Revenue Prediction System...');
    
    // Initialize theme
    initializeTheme();
    
    // Setup event listeners
    initializeEventListeners();
    
    // Initialize chat system
    initializeChatSystem();

    const chatBox = document.getElementById('chatAssistant');
    chatBox.classList.add('hidden');
    
    // Start loading sequence
    startLoadingSequence();
    
    // Initialize default tab content
    switchTab('dashboard');

    const insights = document.getElementById('insightsTab');
    insights.classList.add('hidden');
    
    console.log('Application initialized successfully');
};

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeApplication();
});

// Window Load Event
window.addEventListener('load', () => {
    // Any additional initialization after all resources are loaded
    console.log('All resources loaded');
});

// Handle window resize for responsive charts
window.addEventListener('resize', () => {
    Object.values(appState.charts).forEach(chart => {
        if (chart && typeof chart.resize === 'function') {
            chart.resize();
        }
    });
});

// Prevent form submission on enter key for certain inputs
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.type === 'text' && !e.target.closest('form')) {
        e.preventDefault();
    }
});

// Error handling for uncaught errors
window.addEventListener('error', (e) => {
    console.error('Application error:', e.error);
    showNotification('An unexpected error occurred. Please refresh the page.', 'error');
});

// Service worker registration (if available)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}

// Insight chat box handling function

let isTyping = false;
let messageArea, chatInput, sendButton;

// Initialize chat interface
function initializeBankingChat() {
    messageArea = document.getElementById('chatMessageArea');
    chatInput = document.getElementById('mainChatInput');
    sendButton = document.getElementById('mainChatSend');
    
    setupEventListeners();
    chatInput.focus();
}

// Setup event listeners
function setupEventListeners() {
    // Send button click
    sendButton.addEventListener('click', sendMessage);
    
    // Enter key press
    chatInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
}

// Send message function
function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || isTyping) return;

    // Add user message to chat
    addMessage(message, 'user');
    chatInput.value = '';

    // Show typing indicator
    showTypingIndicator();

    // Make API call
    makeApiCall(message);
}

// Make API call to get response
async function makeApiCall(userMessage) {
    try {
        // Replace with your actual API endpoint
        const response = await fetch('http://localhost:8000/api/chat_insights', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                question: userMessage,
                branchId: appState.selectedBranch ? appState.selectedBranch.id : null,
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Hide typing indicator
        hideTypingIndicator();
        
        // Add bot response
        if (data.message) {
            const messageText = marked(data.message);
            addMessage(messageText, 'bot');

        } else {
            addMessage('I apologize, but I received an empty response. Please try again.', 'bot');
        }

    } catch (error) {
        // Hide typing indicator
        hideTypingIndicator();
        
        // Handle error response
        handleErrorResponse(error);
    }
}

// Handle API error responses
function handleErrorResponse(error) {
    let errorMessage = 'I apologize, but I encountered an issue while processing your request. ';
    
    if (error.message.includes('Failed to fetch')) {
        errorMessage += 'Please check your internet connection and try again.';
    } else if (error.message.includes('500')) {
        errorMessage += 'Our analytics server is temporarily unavailable. Please try again in a moment.';
    } else if (error.message.includes('404')) {
        errorMessage += 'The chat service is temporarily unavailable. Please contact Global IME Bank support if this persists.';
    } else if (error.message.includes('401') || error.message.includes('403')) {
        errorMessage += 'Authentication required. Please refresh the page and try again.';
    } else if (error.message.includes('429')) {
        errorMessage += 'Too many requests. Please wait a moment before trying again.';
    } else if (error.message.includes('timeout')) {
        errorMessage += 'Request timed out. Please try again with a shorter message.';
    } else {
        errorMessage += `Error: ${error.message}`;
    }
    
    addMessage(errorMessage, 'bot');
    
    // Log error for debugging (remove in production)
    console.error('Chat API Error:', error);
}

// Add message to chat area
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `banking-message ${sender}-banking-message`;
    
    const now = new Date();
    const timeString = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    const avatarIcon = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    
    messageDiv.innerHTML = `
        <div class="banking-message-avatar">
            ${avatarIcon}
        </div>
        <div class="banking-message-content">
            <p>${text}</p>
            <span class="banking-message-time">${timeString}</span>
        </div>
    `;
    
    messageArea.appendChild(messageDiv);
    scrollToBottom();
}

// Show typing indicator
function showTypingIndicator() {
    if (isTyping) return;
    
    isTyping = true;
    const typingDiv = document.createElement('div');
    typingDiv.className = 'banking-message bot-banking-message';
    typingDiv.id = 'typingIndicator';
    
    typingDiv.innerHTML = `
        <div class="banking-message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="banking-message-content">
            <p><em>Analyzing your query...</em></p>
        </div>
    `;
    
    messageArea.appendChild(typingDiv);
    scrollToBottom();
}

// Hide typing indicator
function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
    isTyping = false;
}

// Scroll to bottom of chat
function scrollToBottom() {
    setTimeout(function() {
        messageArea.scrollTop = messageArea.scrollHeight;
    }, 100);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeBankingChat();
    console.log('Global IME Bank Analytics Chat initialized');
});


function clearChat() {
    const messageArea = document.getElementById('chatMessageArea');
    if (messageArea) {
        messageArea.innerHTML = ''; // Remove all child message elements
        // Optionally, add a default welcome message after clearing
        addMessage('Hello! I am your Global IME Bank Prediction Analysis Chatbox. Let me help you with analyzing your forcast and preditive analysis.', 'bot');
    }
}


// Export global functions for potential external use
window.GlobalIMEBank = {
    switchTab,
    switchSubtab,
    toggleTheme,
    showNotification,
    formatCurrency,
    formatNumber
};

