const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const translateBtn = document.getElementById('translateBtn');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const resultSection = document.getElementById('resultSection');
const outputText = document.getElementById('outputText');
const charCount = document.getElementById('charCount');
const lineCount = document.getElementById('lineCount');
const loader = document.querySelector('.loader');
const btnText = document.querySelector('.btn-text');

// Handle File Selection
fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        showPreview(file);
    }
});

// Drag and Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        fileInput.files = e.dataTransfer.files;
        showPreview(file);
    }
});

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        previewContainer.classList.remove('hidden');
        translateBtn.disabled = false;
        // Scroll to preview
        previewContainer.scrollIntoView({ behavior: 'smooth' });
    }
    reader.readAsDataURL(file);
}

// Translate Action
translateBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    // UI Feedback
    setLoading(true);
    resultSection.classList.add('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResult(data);
        } else {
            alert('Error: ' + (data.error || 'Gagal memproses gambar'));
        }
    } catch (error) {
        console.error('Fetch error:', error);
        alert('Terjadi kesalahan koneksi ke server.');
    } finally {
        setLoading(false);
    }
});

function setLoading(isLoading) {
    if (isLoading) {
        loader.classList.remove('hidden');
        btnText.textContent = 'Memproses...';
        translateBtn.disabled = true;
    } else {
        loader.classList.add('hidden');
        btnText.textContent = 'Mulai Terjemahkan';
        translateBtn.disabled = false;
    }
}

function displayResult(data) {
    resultSection.classList.remove('hidden');
    outputText.value = data.text;
    
    // Update badges if they exist in data (otherwise use defaults)
    if (data.char_count !== undefined) charCount.textContent = `${data.char_count} Karakter`;
    if (data.line_count !== undefined) lineCount.textContent = `${data.line_count} Baris`;
    
    // Smooth scroll to result
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

// Text to Speech Logic
let speechInstance = null;

// Backend-based TTS Logic
let currentAudio = null;

async function speakText() {
    const text = outputText.value;
    if (!text || !text.trim()) return;

    // If currently playing, stop it
    if (currentAudio && !currentAudio.paused) {
        currentAudio.pause();
        setSpeechActive(false);
        return;
    }

    try {
        setSpeechActive(true);
        const response = await fetch('/speak', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) throw new Error('Gagal mengambil audio dari server');

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        if (currentAudio) {
            currentAudio.pause();
        }
        
        currentAudio = new Audio(url);
        currentAudio.onended = () => setSpeechActive(false);
        currentAudio.onerror = () => {
            setSpeechActive(false);
            alert('Gagal memutar audio dari server.');
        };
        
        currentAudio.play();
    } catch (error) {
        console.error('TTS Error:', error);
        setSpeechActive(false);
        alert('Terjadi kesalahan saat memproses suara.');
    }
}

// Crucial for some browsers: trigger voice loading
if ('speechSynthesis' in window) {
    window.speechSynthesis.getVoices();
    window.speechSynthesis.onvoiceschanged = () => {
        window.speechSynthesis.getVoices();
    };
}

function setSpeechActive(isActive) {
    const speakBtn = document.getElementById('speakBtn');
    if (isActive) {
        speakBtn.classList.add('active');
        speakBtn.innerHTML = '<i class="fas fa-stop"></i> Berhenti';
    } else {
        speakBtn.classList.remove('active');
        speakBtn.innerHTML = '<i class="fas fa-volume-up"></i> Dengarkan';
    }
}

function copyResult() {
    outputText.select();
    document.execCommand('copy');
    
    const copyBtn = document.querySelector('.btn-copy');
    const originalText = copyBtn.innerHTML;
    copyBtn.innerHTML = '<i class="fas fa-check"></i> Tersalin!';
    copyBtn.classList.add('active');
    
    setTimeout(() => {
        copyBtn.innerHTML = originalText;
        copyBtn.classList.remove('active');
    }, 2000);
}
