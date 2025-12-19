let currentMode = 'basic';

function switchMode(mode, evt) {
    if (currentMode === mode) return;
    
    currentMode = mode;
    
    // Update button states
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    if (evt?.target) {
        evt.target.classList.add('active');
    }

    // Get mode sections
    const basicMode = document.getElementById('basic-mode');
    const advancedMode = document.getElementById('advanced-mode');
    
    // Hide all modes
    if (basicMode) basicMode.style.display = 'none';
    if (advancedMode) advancedMode.style.display = 'none';
    
    // Show selected mode
    if (mode === 'basic') {
        if (basicMode) basicMode.style.display = 'block';
        // Initialize with one section if empty
        const sectionsContainer = document.getElementById('sectionsContainer');
        if (sectionsContainer && sectionsContainer.children.length === 0) {
            addSection();
        }
    } else if (mode === 'advanced') {
        if (advancedMode) {
            advancedMode.style.display = 'block';
            advancedMode.classList.add('active');
        }
    }
    
    // Reinitialize custom selects for the new mode
    setTimeout(() => {
        if (typeof initCustomSelects === 'function') {
            initCustomSelects();
        }
    }, 100);
}

function buildPrompt() {
    // Advanced mode uses the freeform prompt
    if (currentMode === 'advanced') {
        const customPrompt = document.getElementById('customPrompt');
        if (customPrompt) {
            return customPrompt.value.trim();
        }
    }
    return '';
}

function showStatus(message, type) {
    const status = document.getElementById('status');
    const statusMessage = status.querySelector('.status-message');
    
    if (statusMessage) {
        statusMessage.innerHTML = message;
    } else {
        status.innerHTML = message;
    }
    
    status.className = `status ${type}`;
    status.style.display = 'block';
    
    // Add/remove generating class on body for heartbeat animation
    if (type === 'loading') {
        document.body.classList.add('generating');
    } else {
        document.body.classList.remove('generating');
    }
}

function hideStatus() {
    document.getElementById('status').style.display = 'none';
}

function showAudioPlayer(downloadUrl, filename) {
    const audioPlayerContainer = document.getElementById('audioPlayerContainer');
    const audioPlayer = document.getElementById('audioPlayer');
    
    // Ensure download URL has /beats prefix if needed
    if (downloadUrl && !downloadUrl.startsWith('http') && !downloadUrl.startsWith('/beats')) {
        if (downloadUrl.startsWith('/')) {
            downloadUrl = '/beats' + downloadUrl;
        } else {
            downloadUrl = '/beats/api/download/' + downloadUrl;
        }
    }
    
    // Clear any previous source
    audioPlayer.pause();
    audioPlayer.src = '';
    audioPlayer.removeAttribute('src');
    
    // Set audio source
    audioPlayer.src = downloadUrl;
    
    // Enable the audio player
    audioPlayer.removeAttribute('disabled');
    audioPlayer.controls = true;
    
    // Add error handling
    audioPlayer.onerror = function(e) {
        console.error('Audio loading error:', e);
        console.error('Failed URL:', downloadUrl);
        console.error('Audio error details:', audioPlayer.error);
        showStatus('❌ Error loading audio file. Please try downloading it instead.', 'error');
    };
    
    audioPlayer.onloadeddata = function() {
        console.log('Audio loaded successfully:', downloadUrl);
        console.log('Audio duration:', audioPlayer.duration);
        console.log('Audio readyState:', audioPlayer.readyState);
    };
    
    audioPlayer.oncanplay = function() {
        console.log('Audio can play now');
    };
    
    // Load the audio
    audioPlayer.load();
    
    // Try to play (will fail if autoplay is blocked, but that's OK)
    audioPlayer.play().catch(function(err) {
        console.log('Autoplay blocked (normal):', err);
    });
    
    // Show the audio player
    audioPlayerContainer.classList.add('active');
    
    // Scroll to top to show the audio player
    window.scrollTo({ top: 0, behavior: 'smooth' });
    
    // Log for debugging
    console.log('Audio player URL:', downloadUrl);
}

function closeAudioPlayer() {
    try {
        const audioPlayerContainer = document.getElementById('audioPlayerContainer');
        const audioPlayer = document.getElementById('audioPlayer');
        
        if (!audioPlayerContainer || !audioPlayer) {
            console.warn('Audio player elements not found');
            return;
        }
        
        // Pause audio if playing (with error handling)
        try {
            if (audioPlayer && !audioPlayer.paused) {
                audioPlayer.pause();
            }
        } catch (e) {
            console.log('Error pausing audio:', e);
        }
        
        // Clear audio source
        try {
            audioPlayer.src = '';
            audioPlayer.removeAttribute('src');
        } catch (e) {
            console.log('Error clearing audio source:', e);
        }
        
        // Hide the audio player
        audioPlayerContainer.classList.remove('active');
    } catch (error) {
        console.error('Error closing audio player:', error);
    }
}

let currentTaskId = null;
let statusCheckInterval = null;

function handleAuthFailure(response) {
    if (response && (response.status === 401 || response.status === 403)) {
        const prefix = window.location.pathname.startsWith('/beats') ? '/beats' : '';
        window.location.href = `${prefix}/login`;
        return true;
    }
    return false;
}

// Real music generation process
async function generateMusic(params) {
    showStatus('🎵 Starting generation...', 'loading');
    
    // Add loading animation to button
    const btn = document.getElementById('createBtn');
    btn.disabled = true;
    btn.classList.add('loading');
    btn.textContent = '🎵 GENERATING...';

    try {
        // Start the generation process
        const response = await fetch('/beats/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params)
        });

        if (handleAuthFailure(response)) return;

        if (response.ok) {
            const result = await response.json();
            if (result.success) {
                currentTaskId = result.task_id;
                showStatus('🎵 Generation started! Loading model...', 'loading');
                
                // Start polling for status updates
                startStatusPolling();
            } else {
                throw new Error(result.error || 'Generation failed to start');
            }
        } else {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Server error');
        }
    } catch (error) {
        showStatus(`❌ Error: ${error.message}`, 'error');
        resetButton();
    }
}

function startStatusPolling() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }

    statusCheckInterval = setInterval(async () => {
        if (!currentTaskId) return;

        try {
            const response = await fetch(`/beats/api/status/${currentTaskId}`);
            if (handleAuthFailure(response)) return;
            if (response.ok) {
                const status = await response.json();
                updateStatus(status);
                
                // Stop polling if completed or failed
                if (status.status === 'success' || status.status === 'error') {
                    clearInterval(statusCheckInterval);
                    statusCheckInterval = null;
                    resetButton();
                }
            }
        } catch (error) {
            console.error('Status check failed:', error);
        }
    }, 2000); // Check every 2 seconds
}

function updateStatus(status) {
    const elapsed = Math.round(status.elapsed_time);
    const modelName = status.model_type === 'musicgen_trained' ? 'MusicGen (Trained)' : 'MusicGen';
    
    switch (status.status) {
        case 'initializing':
            showStatus(`🔧 Initializing ...`, 'loading');
            break;
        case 'connecting':
            showStatus(`🔌 Connecting to  API...`, 'loading');
            break;
        case 'loading_model':
            showStatus(`📦 Loading model...`, 'loading');
            break;
        case 'generating':
            showStatus(`🎵 Generating your beat... This may take a few minutes.`, 'loading');
            break;
        case 'success':
            if (status.download_url) {
                showStatus(`Beat generated! `, 'success');
                // Ensure URL has /beats prefix
                let downloadUrl = status.download_url;
                if (!downloadUrl.startsWith('http') && !downloadUrl.startsWith('/beats')) {
                    downloadUrl = '/beats' + downloadUrl;
                }
                showAudioPlayer(downloadUrl, status.filename);
            } else if (status.filename) {
                // Fallback: construct download URL from filename
                showStatus(`Beat generated!`, 'success');
                showAudioPlayer('/beats/api/download/' + status.filename, status.filename);
            } else {
                showStatus(`Beat generated successfully !`, 'success');
            }
            break;
        case 'error':
            showStatus(`❌ Generation failed: ${status.error}`, 'error');
            break;
        default:
            showStatus(`🎵 Processing... (${elapsed}s elapsed)`, 'loading');
    }
}

function resetButton() {
    const btn = document.getElementById('createBtn');
    btn.disabled = false;
    btn.classList.remove('loading');
    btn.textContent = '🎵 CREATE BEAT';
    currentTaskId = null;
}

document.getElementById('musicForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    hideStatus();

    // Handle new basic mode (composition flow) separately
    if (currentMode === 'basic') {
        generateComposition();
        return;
    }

    const prompt = buildPrompt();
    if (!prompt) {
        showStatus('❌ Please fill in some parameters to generate music', 'error');
        return;
    }

    const params = {
        prompt: prompt,
        duration: currentMode === 'advanced' ? 
            (parseInt(document.getElementById('duration').value) || 15) : 
            (parseInt(document.getElementById('basicDuration').value) || 15),
    };

    // Add model-specific parameters
    if (currentMode === 'advanced') {
        // Advanced mode: always use musicgen with selected checkpoint
        params.modelType = 'musicgen';
        params.checkpointId = document.getElementById('modelSize').value;
    } else {
        // Basic mode defaults to MusicGen with original checkpoint
        params.modelType = 'musicgen';
        params.checkpointId = 'original';
    }

    console.log('Generating music with parameters:', params);
    await generateMusic(params);
});

// Initialize in basic mode - set initial state
document.getElementById('basic-mode').style.display = 'block';
document.getElementById('advanced-mode').style.display = 'none';

// Convert native selects to custom dropdowns
function initCustomSelects() {
    document.querySelectorAll('.form-select').forEach(select => {
        // Skip if already converted
        if (select.closest('.custom-select-container')) return;
        
        // Create custom dropdown structure
        const container = document.createElement('div');
        container.className = 'custom-select-container';
        // Add class based on select ID for easier CSS targeting
        if (select.id) {
            container.classList.add(`select-${select.id}`);
        }
        
        const trigger = document.createElement('div');
        trigger.className = 'custom-select-trigger';
        
        const selectedValue = document.createElement('span');
        selectedValue.className = 'selected-value';
        const placeholder = select.querySelector('option[value=""]');
        selectedValue.textContent = placeholder ? placeholder.textContent : select.options[0].textContent;
        if (placeholder || !select.value) {
            selectedValue.classList.add('placeholder');
        }
        
        const arrow = document.createElement('svg');
        arrow.className = 'arrow';
        arrow.setAttribute('width', '14');
        arrow.setAttribute('height', '14');
        arrow.setAttribute('viewBox', '0 0 14 14');
        arrow.setAttribute('fill', 'none');
        arrow.innerHTML = '<path d="M3.5 5.25L7 8.75L10.5 5.25" stroke="#64ffda" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>';
        
        trigger.appendChild(selectedValue);
        trigger.appendChild(arrow);
        
        const dropdown = document.createElement('div');
        dropdown.className = 'custom-select-dropdown';
        
        // Create options
        Array.from(select.options).forEach((option, index) => {
            const optionDiv = document.createElement('div');
            optionDiv.className = 'custom-select-option';
            optionDiv.textContent = option.textContent;
            optionDiv.dataset.value = option.value;

            if (option.value === select.value) {
                optionDiv.classList.add('selected');
                selectedValue.textContent = option.textContent;
                selectedValue.classList.remove('placeholder');
            }

            optionDiv.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();

                // Set the native select value
                select.value = option.value;

                // Update selected value display
                selectedValue.textContent = option.textContent;
                if (option.value) {
                    selectedValue.classList.remove('placeholder');
                } else {
                    selectedValue.classList.add('placeholder');
                }

                // Update selected option styling
                dropdown.querySelectorAll('.custom-select-option').forEach(opt => {
                    opt.classList.remove('selected');
                });
                optionDiv.classList.add('selected');

                // Trigger change event on native select
                const changeEvent = new Event('change', { bubbles: true });
                select.dispatchEvent(changeEvent);

                // Close dropdown
                trigger.classList.remove('open');
                dropdown.classList.remove('open');
            });

            dropdown.appendChild(optionDiv);
        });
        
        // Toggle dropdown
        trigger.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            const isOpen = trigger.classList.contains('open');
            
            // Close all other dropdowns and return them to original position
            document.querySelectorAll('.custom-select-trigger').forEach(t => {
                if (t !== trigger) {
                    t.classList.remove('open');
                    const nextDropdown = t.nextElementSibling;
                    if (nextDropdown && nextDropdown.classList.contains('custom-select-dropdown')) {
                        nextDropdown.classList.remove('open');
                        // Return dropdown to original position if it was moved to body
                        if (nextDropdown.dataset.originalParent) {
                            const originalParent = document.querySelector(nextDropdown.dataset.originalParent);
                            if (originalParent) {
                                nextDropdown.style.position = '';
                                nextDropdown.style.top = '';
                                nextDropdown.style.left = '';
                                nextDropdown.style.width = '';
                                nextDropdown.style.right = '';
                                nextDropdown.style.zIndex = '';
                                if (nextDropdown._updatePosition) {
                                    window.removeEventListener('scroll', nextDropdown._updatePosition, true);
                                    window.removeEventListener('resize', nextDropdown._updatePosition);
                                    delete nextDropdown._updatePosition;
                                }
                                originalParent.appendChild(nextDropdown);
                                delete nextDropdown.dataset.originalParent;
                            }
                        }
                    }
                }
            });
            
            // Toggle this dropdown
            trigger.classList.toggle('open');
            dropdown.classList.toggle('open');
            
            if (!isOpen) {
                // Move dropdown to body to bypass all stacking context issues
                const rect = container.getBoundingClientRect();
                const scrollY = window.scrollY || window.pageYOffset;
                const scrollX = window.scrollX || window.pageXOffset;
                
                // Store original parent
                if (!container.id) {
                    container.id = 'select-container-' + Math.random().toString(36).substr(2, 9);
                }
                dropdown.dataset.originalParent = '#' + container.id;
                
                // Set fixed positioning relative to viewport
                dropdown.style.position = 'fixed';
                dropdown.style.top = (rect.bottom + scrollY + 6) + 'px';
                dropdown.style.left = (rect.left + scrollX) + 'px';
                dropdown.style.width = rect.width + 'px';
                dropdown.style.right = 'auto';
                dropdown.style.zIndex = '99999';
                
                // Move to body
                document.body.appendChild(dropdown);
                
                // Update position on scroll/resize
                const updatePosition = () => {
                    const newRect = container.getBoundingClientRect();
                    dropdown.style.top = (newRect.bottom + (window.scrollY || window.pageYOffset) + 6) + 'px';
                    dropdown.style.left = (newRect.left + (window.scrollX || window.pageXOffset)) + 'px';
                    dropdown.style.width = newRect.width + 'px';
                };
                
                dropdown._updatePosition = updatePosition;
                window.addEventListener('scroll', updatePosition, true);
                window.addEventListener('resize', updatePosition);
            } else {
                // Return dropdown to original position
                if (dropdown.dataset.originalParent) {
                    const originalParent = document.querySelector(dropdown.dataset.originalParent);
                    if (originalParent) {
                        // Reset styles
                        dropdown.style.position = '';
                        dropdown.style.top = '';
                        dropdown.style.left = '';
                        dropdown.style.width = '';
                        dropdown.style.right = '';
                        dropdown.style.zIndex = '';
                        
                        // Remove event listeners
                        if (dropdown._updatePosition) {
                            window.removeEventListener('scroll', dropdown._updatePosition, true);
                            window.removeEventListener('resize', dropdown._updatePosition);
                            delete dropdown._updatePosition;
                        }
                        
                        // Move back to container
                        originalParent.appendChild(dropdown);
                        delete dropdown.dataset.originalParent;
                    }
                }
            }
        });
        
        // Close on outside click
        const closeHandler = function(e) {
            if (!container.contains(e.target) && !dropdown.contains(e.target)) {
                trigger.classList.remove('open');
                dropdown.classList.remove('open');
                // Return dropdown to original position if it was moved to body
                if (dropdown.dataset.originalParent) {
                    const originalParent = document.querySelector(dropdown.dataset.originalParent);
                    if (originalParent) {
                        dropdown.style.position = '';
                        dropdown.style.top = '';
                        dropdown.style.left = '';
                        dropdown.style.width = '';
                        dropdown.style.right = '';
                        dropdown.style.zIndex = '';
                        originalParent.appendChild(dropdown);
                        delete dropdown.dataset.originalParent;
                    }
                }
            }
        };
        
        // Use capture phase to ensure it works
        document.addEventListener('click', closeHandler, true);
        
        // Wrap select and add custom elements
        select.parentNode.insertBefore(container, select);
        container.appendChild(trigger);
        container.appendChild(dropdown);
        select.classList.add('hidden-select');
        container.appendChild(select);
        
        // Update trigger when native select changes (for form compatibility)
        select.addEventListener('change', function() {
            const selectedOption = select.options[select.selectedIndex];
            selectedValue.textContent = selectedOption.textContent;
            if (select.value) {
                selectedValue.classList.remove('placeholder');
            } else {
                selectedValue.classList.add('placeholder');
            }
            
            // Update selected option in dropdown
            dropdown.querySelectorAll('.custom-select-option').forEach(opt => {
                opt.classList.remove('selected');
                if (opt.dataset.value === select.value) {
                    opt.classList.add('selected');
                }
            });
        });
    });
}

// Initialize custom selects on page load
initCustomSelects();

// Ensure at least one section exists in basic mode on load
document.addEventListener('DOMContentLoaded', () => {
    const sectionsContainer = document.getElementById('sectionsContainer');
    if (sectionsContainer && sectionsContainer.children.length === 0) {
        addSection();
    }
});

// ========== COMPOSITION MODE FUNCTIONS ==========

let sectionCounter = 0;
let compositionTaskId = null;
let compositionStatusInterval = null;

function addSection() {
    sectionCounter++;
    const container = document.getElementById('sectionsContainer');
    const sectionDiv = document.createElement('div');
    sectionDiv.className = 'section-item';
    sectionDiv.draggable = true;
    sectionDiv.dataset.sectionId = sectionCounter;
    // Store original index for syncing with audio results (based on current count)
    const currentSections = container.querySelectorAll('.section-item');
    sectionDiv.dataset.originalIndex = currentSections.length;
    
    sectionDiv.innerHTML = `
        <div class="section-header-row">
            <div class="drag-handle" title="Drag to reorder">☰</div>
            <h3>Section ${sectionCounter}</h3>
            <div style="display: flex; gap: 8px;">
                <button type="button" class="btn-duplicate-section" onclick="duplicateSection(${sectionCounter})" title="Duplicate section">📋</button>
                <button type="button" class="btn-remove-section" onclick="removeSection(${sectionCounter})">×</button>
            </div>
        </div>
        <div class="form-row">
            <div class="form-column form-column-left">
                <div class="form-group">
                    <label>Section Type</label>
                    <select class="section-type form-select" data-section="${sectionCounter}">
                        <option value="Intro">Intro</option>
                        <option value="Middle" selected>Middle</option>
                        <option value="Bridge">Bridge</option>
                        <option value="Drop">Drop</option>
                        <option value="Chorus">Chorus</option>
                        <option value="Ending">Ending</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Bars</label>
                    <input type="number" class="section-bars form-input" min="1" max="32" value="4" data-section="${sectionCounter}">
                </div>
                <div class="form-group">
                    <label>Chord Progression <span class="optional">(optional)</span></label>
                    <input type="text" class="section-chords form-input" placeholder="Cmaj7 - Am7 - Dm7 - G7" data-section="${sectionCounter}">
                </div>
            </div>
            <div class="form-column form-column-right">
                <div class="form-group">
                    <label>Instrumentation <span class="optional">(optional)</span></label>
                    <select class="section-instruments form-select" data-section="${sectionCounter}">
                        <option value="">None</option>
                        <option value="piano">Piano</option>
                        <option value="guitar">Guitar</option>
                        <option value="bass">Bass</option>
                        <option value="drums">Drums</option>
                        <option value="synth">Synth</option>
                        <option value="strings">Strings</option>
                        <option value="brass">Brass</option>
                        <option value="pads">Pads</option>
                        <option value="808">808</option>
                        <option value="hi-hats">Hi-Hats</option>
                        <option value="kick">Kick</option>
                        <option value="snare">Snare</option>
                        <option value="vocal chops">Vocal Chops</option>
                        <option value="rhodes">Rhodes</option>
                        <option value="organ">Organ</option>
                        <option value="piano, bass, drums">Piano, Bass, Drums</option>
                        <option value="synth, pads, 808">Synth, Pads, 808</option>
                        <option value="guitar, bass, drums">Guitar, Bass, Drums</option>
                        <option value="strings, pads">Strings, Pads</option>
                        <option value="808, hi-hats, kick, snare">808, Hi-Hats, Kick, Snare</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Pitch Curve</label>
                    <select class="section-pitch form-select" data-section="${sectionCounter}">
                        <option value="stable" selected>Stable</option>
                        <option value="rising">Rising</option>
                        <option value="falling">Falling</option>
                        <option value="stable → slight rise at end">Stable → Slight Rise</option>
                        <option value="stable → slight fall at end">Stable → Slight Fall</option>
                    </select>
                </div>
            </div>
        </div>
    `;
    
    container.appendChild(sectionDiv);
    
    // Initialize custom selects for ALL dropdowns in the new section
    // This ensures style, preset, instrumentation, section-type, and pitch all get custom styling
    setTimeout(() => {
        if (typeof initCustomSelects === 'function') {
            initCustomSelects(); // Will convert all .form-select elements, skipping already converted ones
        }
    }, 100);
}

function removeSection(sectionId) {
    const section = document.querySelector(`[data-section-id="${sectionId}"]`);
    if (section) {
        // Get the original index to find the corresponding audio
        const originalIndex = section.dataset.originalIndex;
        
        // Remove the section
        section.remove();
        
        // Find and remove the corresponding audio result
        if (originalIndex !== undefined) {
            const audioContainer = document.getElementById('sectionsResults');
            if (audioContainer) {
                const audioResult = audioContainer.querySelector(`[data-original-index="${originalIndex}"]`);
                if (audioResult) {
                    audioResult.remove();
                }
            }
        }
        
        // Hide results container if no sections remain
        const remainingSections = document.querySelectorAll('.section-item');
        const resultsContainer = document.getElementById('sectionsResultsContainer');
        if (remainingSections.length === 0 && resultsContainer) {
            resultsContainer.style.display = 'none';
        }
    }
}

function duplicateSection(sectionId) {
    const sourceSection = document.querySelector(`[data-section-id="${sectionId}"]`);
    if (!sourceSection) return;
    
    // Get all values from source section
    const sectionType = sourceSection.querySelector('.section-type').value;
    const bars = sourceSection.querySelector('.section-bars').value;
    const chords = sourceSection.querySelector('.section-chords').value;
    const instruments = sourceSection.querySelector('.section-instruments').value;
    const pitch = sourceSection.querySelector('.section-pitch').value;
    
    // Create new section
    sectionCounter++;
    const container = document.getElementById('sectionsContainer');
    const sectionDiv = document.createElement('div');
    sectionDiv.className = 'section-item';
    sectionDiv.draggable = true;
    sectionDiv.dataset.sectionId = sectionCounter;
    // Store original index for syncing with audio results (based on current count)
    const currentSections = container.querySelectorAll('.section-item');
    sectionDiv.dataset.originalIndex = currentSections.length;
    
    // Build HTML with duplicated values - use the same structure as addSection
    sectionDiv.innerHTML = `
        <div class="section-header-row">
            <div class="drag-handle" title="Drag to reorder">☰</div>
            <h3>Section ${sectionCounter}</h3>
            <div style="display: flex; gap: 8px;">
                <button type="button" class="btn-duplicate-section" onclick="duplicateSection(${sectionCounter})" title="Duplicate section">📋</button>
                <button type="button" class="btn-remove-section" onclick="removeSection(${sectionCounter})">×</button>
            </div>
        </div>
        <div class="form-row">
            <div class="form-column form-column-left">
                <div class="form-group">
                    <label>Section Type</label>
                    <select class="section-type form-select" data-section="${sectionCounter}">
                        <option value="Intro" ${sectionType === 'Intro' ? 'selected' : ''}>Intro</option>
                        <option value="Middle" ${sectionType === 'Middle' ? 'selected' : ''}>Middle</option>
                        <option value="Bridge" ${sectionType === 'Bridge' ? 'selected' : ''}>Bridge</option>
                        <option value="Drop" ${sectionType === 'Drop' ? 'selected' : ''}>Drop</option>
                        <option value="Chorus" ${sectionType === 'Chorus' ? 'selected' : ''}>Chorus</option>
                        <option value="Ending" ${sectionType === 'Ending' ? 'selected' : ''}>Ending</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Bars</label>
                    <input type="number" class="section-bars form-input" min="1" max="32" value="${bars}" data-section="${sectionCounter}">
                </div>
                <div class="form-group">
                    <label>Chord Progression <span class="optional">(optional)</span></label>
                    <input type="text" class="section-chords form-input" placeholder="Cmaj7 - Am7 - Dm7 - G7" value="${chords || ''}" data-section="${sectionCounter}">
                </div>
            </div>
            <div class="form-column form-column-right">
                <div class="form-group">
                    <label>Instrumentation <span class="optional">(optional)</span></label>
                    <select class="section-instruments form-select" data-section="${sectionCounter}">
                        <option value="">None</option>
                        <option value="piano" ${instruments === 'piano' ? 'selected' : ''}>Piano</option>
                        <option value="guitar" ${instruments === 'guitar' ? 'selected' : ''}>Guitar</option>
                        <option value="bass" ${instruments === 'bass' ? 'selected' : ''}>Bass</option>
                        <option value="drums" ${instruments === 'drums' ? 'selected' : ''}>Drums</option>
                        <option value="synth" ${instruments === 'synth' ? 'selected' : ''}>Synth</option>
                        <option value="strings" ${instruments === 'strings' ? 'selected' : ''}>Strings</option>
                        <option value="brass" ${instruments === 'brass' ? 'selected' : ''}>Brass</option>
                        <option value="pads" ${instruments === 'pads' ? 'selected' : ''}>Pads</option>
                        <option value="808" ${instruments === '808' ? 'selected' : ''}>808</option>
                        <option value="hi-hats" ${instruments === 'hi-hats' ? 'selected' : ''}>Hi-Hats</option>
                        <option value="kick" ${instruments === 'kick' ? 'selected' : ''}>Kick</option>
                        <option value="snare" ${instruments === 'snare' ? 'selected' : ''}>Snare</option>
                        <option value="vocal chops" ${instruments === 'vocal chops' ? 'selected' : ''}>Vocal Chops</option>
                        <option value="rhodes" ${instruments === 'rhodes' ? 'selected' : ''}>Rhodes</option>
                        <option value="organ" ${instruments === 'organ' ? 'selected' : ''}>Organ</option>
                        <option value="piano, bass, drums" ${instruments === 'piano, bass, drums' ? 'selected' : ''}>Piano, Bass, Drums</option>
                        <option value="synth, pads, 808" ${instruments === 'synth, pads, 808' ? 'selected' : ''}>Synth, Pads, 808</option>
                        <option value="guitar, bass, drums" ${instruments === 'guitar, bass, drums' ? 'selected' : ''}>Guitar, Bass, Drums</option>
                        <option value="strings, pads" ${instruments === 'strings, pads' ? 'selected' : ''}>Strings, Pads</option>
                        <option value="808, hi-hats, kick, snare" ${instruments === '808, hi-hats, kick, snare' ? 'selected' : ''}>808, Hi-Hats, Kick, Snare</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Pitch Curve</label>
                    <select class="section-pitch form-select" data-section="${sectionCounter}">
                        <option value="stable" ${pitch === 'stable' ? 'selected' : ''}>Stable</option>
                        <option value="rising" ${pitch === 'rising' ? 'selected' : ''}>Rising</option>
                        <option value="falling" ${pitch === 'falling' ? 'selected' : ''}>Falling</option>
                        <option value="stable → slight rise at end" ${pitch === 'stable → slight rise at end' ? 'selected' : ''}>Stable → Slight Rise</option>
                        <option value="stable → slight fall at end" ${pitch === 'stable → slight fall at end' ? 'selected' : ''}>Stable → Slight Fall</option>
                    </select>
                </div>
            </div>
        </div>
    `;
    
    // Insert after the source section
    sourceSection.parentNode.insertBefore(sectionDiv, sourceSection.nextSibling);
    
    // Copy generated music if it exists for the source section
    const audioContainer = document.getElementById('sectionsResults');
    const resultsContainer = document.getElementById('sectionsResultsContainer');
    if (audioContainer && resultsContainer) {
        // Prefer matching by generated filename (stable across re-renders)
        const sourceFilename =
            sourceSection.dataset.generatedFilename ||
            sourceSection.dataset.sectionFilename ||
            '';

        let sourceAudioResult = null;
        if (sourceFilename) {
            sourceAudioResult = audioContainer.querySelector(`[data-section-filename="${CSS.escape(sourceFilename)}"]`);
        }

        // Fallback to matching by original index
        if (!sourceAudioResult) {
            const sourceOriginalIndex = sourceSection.dataset.originalIndex;
            if (sourceOriginalIndex !== undefined) {
                sourceAudioResult = audioContainer.querySelector(`[data-original-index="${sourceOriginalIndex}"]`);
            }
        }

        if (sourceAudioResult) {
                // Show results container if it's hidden
                resultsContainer.style.display = 'block';
                
                // Clone the audio result
                const newAudioResult = sourceAudioResult.cloneNode(true);
                const newOriginalIndex = sectionDiv.dataset.originalIndex;
                
                // Update the cloned audio result with new index
                newAudioResult.dataset.originalIndex = newOriginalIndex;
                newAudioResult.dataset.sectionIndex = audioContainer.querySelectorAll('.section-result').length;
                // Mark as a UI clone so re-renders preserve it
                newAudioResult.dataset.isClone = "true";
                // Re-attach this clone after its source section on re-render
                newAudioResult.dataset.insertAfterFilename = sourceAudioResult.dataset.sectionFilename || "";
                
                // Update the section name in the cloned result to match the new section
                const sectionNameElement = newAudioResult.querySelector('h4');
                if (sectionNameElement) {
                    sectionNameElement.textContent = `${sectionType} Section`;
                }
                
                // Insert the cloned audio result after the source audio result
                sourceAudioResult.parentNode.insertBefore(newAudioResult, sourceAudioResult.nextSibling);
                
                // Reinitialize drag and drop for the new audio result
                initGeneratedSectionDragAndDrop();
        }
    }
    
    // Initialize custom selects for the duplicated section
    setTimeout(() => {
        if (typeof initCustomSelects === 'function') {
            initCustomSelects();
        }
    }, 100);
}

function collectSections() {
    const sections = [];
    const sectionItems = document.querySelectorAll('.section-item');
    
    sectionItems.forEach(item => {
        const sectionId = item.dataset.sectionId;
        
        // Get values from native select elements (which are synced by custom dropdowns if present)
        const sectionTypeSelect = item.querySelector('.section-type');
        const sectionPitchSelect = item.querySelector('.section-pitch');
        
        const instrumentsSelect = item.querySelector('.section-instruments');
        const instrumentationValue = instrumentsSelect ? instrumentsSelect.value : '';
        const instrumentation = instrumentationValue ? [instrumentationValue] : [];
        
        const section = {
            name: sectionTypeSelect ? sectionTypeSelect.value : 'Middle',
            bars: parseInt(item.querySelector('.section-bars').value) || 4,
            chord_progression: item.querySelector('.section-chords').value.trim(),
            instrumentation: instrumentation,
            pitch_curve: sectionPitchSelect ? sectionPitchSelect.value : 'stable'
        };
        
        // Log for debugging - ensure selected values are captured
        console.log(`Section ${sectionId} collected:`, section);
        
        sections.push(section);
    });
    
    console.log('All sections collected:', sections);
    return sections;
}

async function generateComposition() {
    const sections = collectSections();
    
    if (sections.length === 0) {
        showStatus('❌ Please add at least one section', 'error');
        return;
    }
    
    showStatus('🎵 Starting composition generation...', 'loading');
    
    const btn = document.getElementById('createBtn');
    btn.disabled = true;
    btn.classList.add('loading');
    btn.textContent = '🎵 GENERATING COMPOSITION...';
    
    try {
        const globalTempo = parseFloat(document.getElementById('globalTempo').value) || 120;
        const globalKey = document.getElementById('globalKey').value.trim() || 'C major';
        const globalStyle = document.getElementById('globalStyle').value || '';
        const globalPreset = document.getElementById('globalPreset').value || '';
        const checkpointId = document.getElementById('compositionModelSize').value;
        
        // Combine global style/preset into a single base style string
        let baseStyle = '';
        if (globalStyle) baseStyle = globalStyle;
        if (globalPreset) baseStyle += baseStyle ? `, ${globalPreset}` : globalPreset;
        
        // Use music-chain model (mapped to trained checkpoint on backend)
        const response = await fetch('/beats/api/generate-sections', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sections: sections,
                tempo: globalTempo,
                key: globalKey,
                base_style: baseStyle,
                modelType: 'music-chain',
                checkpointId: checkpointId
                // Do NOT send combineTrack or shuffle - sections are generated individually
            })
        });
        
        if (handleAuthFailure(response)) return;

        if (response.ok) {
            const result = await response.json();
            if (result.success) {
                compositionTaskId = result.task_id;
                // Store task_id in DOM for persistence
                const resultsContainer = document.getElementById('sectionsResultsContainer');
                if (resultsContainer) {
                    resultsContainer.dataset.taskId = compositionTaskId;
                }
                console.log('Composition task ID set:', compositionTaskId);
                showStatus(`🎵 Composition generation started! Generating ${sections.length} sections...`, 'loading');
                startCompositionStatusPolling();
            } else {
                throw new Error(result.error || 'Composition generation failed to start');
            }
        } else {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Server error');
        }
    } catch (error) {
        showStatus(`❌ Error: ${error.message}`, 'error');
        resetButton();
    }
}

function startCompositionStatusPolling() {
    if (compositionStatusInterval) {
        clearInterval(compositionStatusInterval);
    }
    
    compositionStatusInterval = setInterval(async () => {
        if (!compositionTaskId) return;
        
        try {
            const response = await fetch(`/beats/api/sections-status/${compositionTaskId}`);
            if (handleAuthFailure(response)) return;
            if (response.ok) {
                const status = await response.json();
                updateCompositionStatus(status);
                
                if (status.status === 'success' || status.status === 'error') {
                    clearInterval(compositionStatusInterval);
                    compositionStatusInterval = null;
                    resetButton();
                }
            }
        } catch (error) {
            console.error('Composition status check failed:', error);
        }
    }, 3000); // Check every 3 seconds
}

function updateCompositionStatus(status) {
    const elapsed = Math.round(status.elapsed_time);
    
    if (status.status === 'generating') {
        const progress = status.generated_sections > 0 
            ? ` (${status.generated_sections}/${status.total_sections} sections complete)`
            : '';
        showStatus(`🎵 Generating sections...${progress}`, 'loading');
        
        // Display sections as they're generated
        displaySectionResults(status.sections || []);
    } else if (status.status === 'success') {
        showStatus(`✅ All sections generated! ${status.total_sections} sections ready.`, 'success');
        
        // Display all section results
        if (status.sections && status.sections.length > 0) {
            displaySectionResults(status.sections);
            
            // Show combine button if we have 2+ sections
            if (status.sections.length >= 2) {
                document.getElementById('combineSectionContainer').style.display = 'block';
            }
            
                // If combined track already exists (from previous combine), show it
            if (status.combined_track) {
                setTimeout(() => {
                    showAudioPlayer(status.combined_track.audio_url, status.combined_track.filename);
                }, 1000);
            }
        }
    } else if (status.status === 'error') {
        showStatus(`❌ Composition generation failed: ${status.error}`, 'error');
    }
}

function displaySectionResults(sections) {
    const container = document.getElementById('sectionsResults');
    const resultsContainer = document.getElementById('sectionsResultsContainer');
    
    if (!container || !resultsContainer) return;
    
    // Show results container
    resultsContainer.style.display = 'block';
    
    // Preserve any UI-cloned results (e.g., when user duplicates a section and wants to copy audio)
    // The status poll re-renders this container frequently, so we must re-attach clones.
    const existingClones = Array.from(container.querySelectorAll('.section-result[data-is-clone="true"]'));
    const clonesByAfterFilename = new Map();
    existingClones.forEach((cloneEl) => {
        const after = cloneEl.dataset.insertAfterFilename || '';
        if (!clonesByAfterFilename.has(after)) clonesByAfterFilename.set(after, []);
        clonesByAfterFilename.get(after).push(cloneEl);
    });

    // Clear existing results (server-driven list will be rebuilt below)
    container.innerHTML = '';
    
    // Add each section with its audio player (make draggable)
    sections.forEach((section, idx) => {
        const sectionDiv = document.createElement('div');
        sectionDiv.className = 'section-result';
        sectionDiv.draggable = true;
        sectionDiv.dataset.sectionFilename = section.filename;
        sectionDiv.dataset.sectionIndex = idx;
        // Store original section ID for matching with form sections
        sectionDiv.dataset.originalIndex = idx;
        sectionDiv.style.cssText = 'margin-bottom: 20px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 8px; cursor: move; position: relative;';
        
        sectionDiv.innerHTML = `
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 20px; margin-right: 10px; color: #888; cursor: move;">☰</span>
                <h4 style="margin: 0; color: #4CAF50; flex: 1;">${section.name} Section</h4>
            </div>
            <p style="margin: 5px 0; color: #aaa; font-size: 0.9em;">
                ${section.bars} bars • ${section.duration.toFixed(1)}s • 
                ${section.audio_parameters.tempo} BPM • ${section.audio_parameters.key}
            </p>
            <audio controls style="width: 100%; margin-top: 10px;" src="${section.audio_url}"></audio>
        `;
        
        container.appendChild(sectionDiv);

        // Persist generated filename onto the matching form section so duplicate/remove can find audio reliably
        const formContainer = document.getElementById('sectionsContainer');
        if (formContainer) {
            const matchingFormSection = formContainer.querySelector(`.section-item[data-original-index="${idx}"]`);
            if (matchingFormSection) {
                matchingFormSection.dataset.generatedFilename = section.filename || '';
                matchingFormSection.dataset.sectionFilename = section.filename || '';
            }
        }

        // Re-attach any clones that should appear immediately after this generated section
        const afterFilename = section.filename || '';
        const clonesForThis = clonesByAfterFilename.get(afterFilename);
        if (clonesForThis && clonesForThis.length) {
            clonesForThis.forEach((cloneEl) => container.appendChild(cloneEl));
            clonesByAfterFilename.delete(afterFilename);
        }
    });

    // Append any remaining clones (fallback)
    clonesByAfterFilename.forEach((cloneEls) => {
        cloneEls.forEach((cloneEl) => container.appendChild(cloneEl));
    });
    
    // Initialize drag and drop for generated sections
    initGeneratedSectionDragAndDrop();
}

// Sync audio results order to match form sections order
function syncAudioResultsToFormSections() {
    const formContainer = document.getElementById('sectionsContainer');
    const audioContainer = document.getElementById('sectionsResults');
    
    if (!formContainer || !audioContainer) return;
    
    // Get form sections and audio results in current DOM order
    const formSections = Array.from(formContainer.querySelectorAll('.section-item'));
    const audioResults = Array.from(audioContainer.querySelectorAll('.section-result'));
    
    if (formSections.length !== audioResults.length || formSections.length === 0) return;
    
    // Get form sections' original indices (stored when created)
    const formOriginalIndices = formSections.map(form => 
        parseInt(form.dataset.originalIndex || 0)
    );
    
    // Create mapping: original index -> audio result element
    const audioMap = new Map();
    audioResults.forEach(audio => {
        const originalIdx = parseInt(audio.dataset.originalIndex || 0);
        audioMap.set(originalIdx, audio);
    });
    
    // Reorder audio results to match form sections' new order
    // Form section at position 0 (with original index X) -> Audio result with original index X
    const reorderedAudio = [];
    formOriginalIndices.forEach(originalIdx => {
        const audioResult = audioMap.get(originalIdx);
        if (audioResult) {
            reorderedAudio.push(audioResult);
        }
    });
    
    // Apply new order to DOM
    reorderedAudio.forEach((audioResult, newIdx) => {
        const currentElement = audioContainer.children[newIdx];
        if (currentElement !== audioResult) {
            audioContainer.insertBefore(audioResult, currentElement || null);
        }
    });
}

// Sync form sections order to match audio results order
function syncFormSectionsToAudioResults() {
    const formContainer = document.getElementById('sectionsContainer');
    const audioContainer = document.getElementById('sectionsResults');
    
    if (!formContainer || !audioContainer) return;
    
    // Get audio results and form sections in current DOM order
    const audioResults = Array.from(audioContainer.querySelectorAll('.section-result'));
    const formSections = Array.from(formContainer.querySelectorAll('.section-item'));
    
    if (formSections.length !== audioResults.length || formSections.length === 0) return;
    
    // Get audio results' original indices
    const audioOriginalIndices = audioResults.map(audio => 
        parseInt(audio.dataset.originalIndex || 0)
    );
    
    // Create mapping: original index -> form section element
    const formMap = new Map();
    formSections.forEach(form => {
        const originalIdx = parseInt(form.dataset.originalIndex || 0);
        formMap.set(originalIdx, form);
    });
    
    // Reorder form sections to match audio results' new order
    // Audio result at position 0 (with original index X) -> Form section with original index X
    const reorderedForms = [];
    audioOriginalIndices.forEach(originalAudioIdx => {
        const formSection = formMap.get(originalAudioIdx);
        if (formSection) {
            reorderedForms.push(formSection);
        }
    });
    
    // Apply new order to DOM
    reorderedForms.forEach((formSection, newIdx) => {
        const currentElement = formContainer.children[newIdx];
        if (currentElement !== formSection) {
            formContainer.insertBefore(formSection, currentElement || null);
        }
    });
}

// Simple drag-and-drop reordering for sections
function initSectionDragAndDrop() {
    const container = document.getElementById('sectionsContainer');
    if (!container) return;
    
    let dragSrcEl = null;
    
    const handleDragStart = (e) => {
        if (!e.target.classList.contains('section-item')) return;
        dragSrcEl = e.target;
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/html', e.target.outerHTML);
        e.target.classList.add('dragging');
    };
    
    const handleDragOver = (e) => {
        if (e.preventDefault) e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        const target = e.target.closest('.section-item');
        if (target && target !== dragSrcEl) {
            const bounding = target.getBoundingClientRect();
            const offset = bounding.y + (bounding.height / 2);
            if (e.clientY - offset > 0) {
                target.after(dragSrcEl);
            } else {
                target.before(dragSrcEl);
            }
        }
        return false;
    };
    
    const handleDrop = (e) => {
        if (e.stopPropagation) e.stopPropagation();
        dragSrcEl?.classList.remove('dragging');
        dragSrcEl = null;
        // Sync audio results to match form sections order
        syncAudioResultsToFormSections();
        return false;
    };
    
    const handleDragEnd = (e) => {
        e.target.classList.remove('dragging');
        // Sync audio results to match form sections order
        syncAudioResultsToFormSections();
    };
    
    container.addEventListener('dragstart', handleDragStart);
    container.addEventListener('dragover', handleDragOver);
    container.addEventListener('drop', handleDrop);
    container.addEventListener('dragend', handleDragEnd);
}

// Initialize drag-and-drop once the DOM is ready
document.addEventListener('DOMContentLoaded', initSectionDragAndDrop);

// Drag and drop for generated section results
function initGeneratedSectionDragAndDrop() {
    const container = document.getElementById('sectionsResults');
    if (!container) return;
    
    let dragSrcEl = null;
    
    const handleDragStart = (e) => {
        if (!e.target.classList.contains('section-result') && !e.target.closest('.section-result')) return;
        dragSrcEl = e.target.classList.contains('section-result') ? e.target : e.target.closest('.section-result');
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/html', dragSrcEl.outerHTML);
        dragSrcEl.classList.add('dragging');
        dragSrcEl.style.opacity = '0.5';
    };
    
    const handleDragOver = (e) => {
        if (e.preventDefault) e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        const target = e.target.closest('.section-result');
        if (target && target !== dragSrcEl) {
            const bounding = target.getBoundingClientRect();
            const offset = bounding.y + (bounding.height / 2);
            if (e.clientY - offset > 0) {
                target.after(dragSrcEl);
            } else {
                target.before(dragSrcEl);
            }
        }
        return false;
    };
    
    const handleDrop = (e) => {
        if (e.stopPropagation) e.stopPropagation();
        if (dragSrcEl) {
            dragSrcEl.classList.remove('dragging');
            dragSrcEl.style.opacity = '1';
        }
        dragSrcEl = null;
        // Sync form sections to match audio results order
        syncFormSectionsToAudioResults();
        return false;
    };
    
    const handleDragEnd = (e) => {
        if (e.target.classList.contains('section-result')) {
            e.target.classList.remove('dragging');
            e.target.style.opacity = '1';
        }
        // Sync form sections to match audio results order
        syncFormSectionsToAudioResults();
    };
    
    // Remove old listeners if any
    container.removeEventListener('dragstart', handleDragStart);
    container.removeEventListener('dragover', handleDragOver);
    container.removeEventListener('drop', handleDrop);
    container.removeEventListener('dragend', handleDragEnd);
    
    // Add new listeners
    container.addEventListener('dragstart', handleDragStart);
    container.addEventListener('dragover', handleDragOver);
    container.addEventListener('drop', handleDrop);
    container.addEventListener('dragend', handleDragEnd);
}

async function combineSections() {
    // Try to get task_id from multiple sources
    let taskId = compositionTaskId;
    
    // If not in variable, try to get from DOM
    if (!taskId) {
        const resultsContainer = document.getElementById('sectionsResultsContainer');
        if (resultsContainer && resultsContainer.dataset.taskId) {
            taskId = resultsContainer.dataset.taskId;
            compositionTaskId = taskId; // Restore to variable
            console.log('Restored task ID from DOM:', taskId);
        }
    }
    
    if (!taskId) {
        showStatus('❌ No composition task found. Please generate sections first.', 'error');
        return;
    }
    
    console.log('Combining sections with task ID:', taskId);
    
    const btn = document.getElementById('combineSectionsBtn');
    btn.disabled = true;
    btn.textContent = '🎵 COMBINING...';
    
    showStatus('🎵 Combining sections into full track...', 'loading');
    
    try {
        // Get sections in their current UI order (from drag and drop)
        const container = document.getElementById('sectionsResults');
        const sectionOrder = [];
        if (container) {
            const sectionElements = container.querySelectorAll('.section-result');
            sectionElements.forEach(el => {
                const filename = el.dataset.sectionFilename;
                if (filename) {
                    sectionOrder.push(filename);
                }
            });
        }
        
        console.log('Section order:', sectionOrder);
        
        const response = await fetch(`/beats/api/combine-sections/${taskId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                section_order: sectionOrder.length > 0 ? sectionOrder : null
            })
        });
        
        if (handleAuthFailure(response)) return;

        if (response.ok) {
            const result = await response.json();
            if (result.success) {
                showStatus('✅ Sections combined successfully!', 'success');
                
                // Show combined track in audio player
                setTimeout(() => {
                    showAudioPlayer(result.audio_url, result.filename);
                }, 500);
                
                // Update status to refresh and show combined track
                setTimeout(() => {
                    fetch(`/beats/api/sections-status/${taskId}`)
                        .then(r => {
                            if (handleAuthFailure(r)) return null;
                            return r.json();
                        })
                        .then(status => {
                            if (status) updateCompositionStatus(status);
                        });
                }, 1000);
            } else {
                throw new Error(result.error || 'Combination failed');
            }
        } else {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Server error');
        }
    } catch (error) {
        showStatus(`❌ Error combining sections: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '🎵 COMBINE SECTIONS';
    }
}


// Add mouse tracking for form-section hover effect (follows pointer)
document.querySelectorAll('.form-section').forEach(section => {
    section.addEventListener('mousemove', function(e) {
        const rect = this.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        this.style.setProperty('--mouse-x', x + '%');
        this.style.setProperty('--mouse-y', y + '%');
    });
});

