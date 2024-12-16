// Calculate image dimensions to fit within the canvas
function getCanvasImageDimensions(canvas, image) {
    let drawWidth, drawHeight;
    const aspectRatio = image.width / image.height;

    if (aspectRatio > 1) {
        drawWidth = canvas.width;
        drawHeight = canvas.width / aspectRatio;
    } else {
        drawHeight = canvas.height;
        drawWidth = canvas.height * aspectRatio;
    }

    const offsetX = (canvas.width - drawWidth) / 2;
    const offsetY = (canvas.height - drawHeight) / 2;
    return { drawWidth, drawHeight, offsetX, offsetY };
}

// Handle drag events for styling dropzone
function handleDragOver(e) {
    e.preventDefault();
}
function handleDragEnter(e) {
    e.preventDefault();
    document.getElementById('dropzone').classList.add('bg-gray-100', 'border-blue-500');
}
function handleDragLeave(e) {
    e.preventDefault();
    document.getElementById('dropzone').classList.remove('bg-gray-100', 'border-blue-500');
}

// Handle file drop onto dropzone
function handleDrop(e) {
    e.preventDefault();
    const fileInput = document.getElementById('dropzone-file');
    const droppedFiles = e.dataTransfer.files;

    if (droppedFiles.length) {
        const file = droppedFiles[0];
        if (!isValidDicom(file)) {
            alert('Only .dicom files are allowed!');
            return;
        }
        fileInput.files = droppedFiles;
        document.getElementById('upload-form').dispatchEvent(new Event('submit'));
    }
    document.getElementById('dropzone').classList.remove('bg-gray-100', 'border-blue-500');
}

// Set up the file input submission with validation
function setupFileInput() {
    document.getElementById('upload-form').onsubmit = handleFileUpload;
    document.getElementById('dropzone-file').addEventListener('change', () => {
        document.getElementById('upload-form').dispatchEvent(new Event('submit'));
    });
}

// Set up dropzone events
function setupDropzone() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('dropzone-file');

    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', handleDragOver);
    dropzone.addEventListener('dragenter', handleDragEnter);
    dropzone.addEventListener('dragleave', handleDragLeave);
    dropzone.addEventListener('drop', handleDrop);
}

// Set up the reupload button
function setupReuploadButton() {
    document.getElementById('reupload-button').onclick = resetCanvas;
}

function showLoadingModal() {
    document.getElementById('loading-modal').classList.remove('hidden');
}

function hideLoadingModal() {
    document.getElementById('loading-modal').classList.add('hidden');
}

// Handle file upload
async function handleFileUpload(e) {
    e.preventDefault();
    const fileInput = document.getElementById('dropzone-file');
    if (!fileInput.files.length) return;

    const file = fileInput.files[0];
    if (!isValidDicom(file)) {
        alert('Only (.dcm, .dcim, .dc3, .dic, .dicom) files are allowed!');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        // Show the modal when the request starts
        showLoadingModal();
        const response = await fetch('/demo', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (data.original_image && data.prediction && data.table) {
            displayImagesAndTable(data);
        }
    } catch (error) {
        console.error('Error uploading file:', error);
        alert('Failed to upload the file. Please try again.');
    } 
    // Hide the modal when the request finishes
    finally {
        hideLoadingModal();
    }
}

// Validate dicom files
function isValidDicom(file) {
    const validExtensions = ['.dcm', '.dcim', '.dc3', '.dic', '.dicom'];
    const fileName = file.name.toLowerCase();
    return validExtensions.some(extension => fileName.endsWith(extension));
}

// Draw bounding boxes onto the prediction canvas
function drawBoundingBoxes(ctx, canvas, shapes, image) {
    // Get the scaled image dimensions and offsets
    const { drawWidth, drawHeight, offsetX, offsetY } = getCanvasImageDimensions(canvas, image);

    // Scale factor between original image and drawn image
    const scaleX = drawWidth / image.width;
    const scaleY = drawHeight / image.height;

    shapes.forEach((shape) => {
        const { points, label } = shape;

        // Scale and translate bounding box coordinates to canvas space
        const x = points[0].x * scaleX + offsetX; // Adjust X for scaling and offset
        const y = points[0].y * scaleY + offsetY; // Adjust Y for scaling and offset
        const width = (points[1].x - points[0].x) * scaleX; // Scale width
        const height = (points[2].y - points[0].y) * scaleY; // Scale height

        // Draw the bounding box
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        // Add the label
        ctx.font = '14px Arial';
        ctx.fillStyle = 'red';
        ctx.fillText(label, x, y - 5); // Draw the label above the bounding box
    });
}

// Display images and table
function displayImagesAndTable(data) {
    const originalCanvas = document.getElementById('canvas-original');
    const predictionCanvas = document.getElementById('canvas-prediction');
    const ctxOriginal = originalCanvas.getContext('2d');
    const ctxPrediction = predictionCanvas.getContext('2d');

    const originalImage = new Image();
    const predictionImage = new Image();

    predictionImage.onload = () => drawImage(predictionCanvas, ctxPrediction, predictionImage);
    originalImage.onload = () => {
        drawImage(originalCanvas, ctxOriginal, originalImage);
        drawBoundingBoxes(ctxOriginal, originalCanvas, data.shapes, originalImage); // Draw bounding boxes
    };

    originalImage.src = data.original_image;
    predictionImage.src = data.prediction;

    displayDataTable(data.table);
    showCanvasElements();
}

// Draw image onto canvas
function drawImage(canvas, ctx, image) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const dimensions = getCanvasImageDimensions(canvas, image);
    ctx.drawImage(image, dimensions.offsetX, dimensions.offsetY, dimensions.drawWidth, dimensions.drawHeight);
}

function displayDataTable(dataArray) {
    const tableContainer = document.getElementById('data-table');
    const table = document.createElement('table');
    table.classList.add('table-auto', 'border-collapse', 'w-full', 'border', 'bg-white', 'rounded');

    // Create the header row
    const headerRow = document.createElement('tr');

    // Table columns
    const headers = ['Attribute', 'Value'];
    headers.forEach(headerText => {
        const th = document.createElement('th');
        th.textContent = headerText;
        th.classList.add('border', 'p-2', 'text-left', 'bg-gray-200');
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    // Create data rows
    dataArray.forEach(item => {
        const row = document.createElement('tr');
        const nameCell = document.createElement('td');
        nameCell.textContent = item.name;
        nameCell.classList.add('border', 'p-2');
        row.appendChild(nameCell);

        const valueCell = document.createElement('td');
        valueCell.textContent = item.value;
        valueCell.classList.add('border', 'p-2');
        row.appendChild(valueCell);

        table.appendChild(row);
    });

    // Clear previous content and append the new table
    tableContainer.innerHTML = '';
    tableContainer.appendChild(table);
    tableContainer.classList.remove('hidden');
}

// Show elements
function showCanvasElements() {
    document.getElementById('canvas-original').classList.remove('hidden');
    document.getElementById('canvas-prediction').classList.remove('hidden');
    document.getElementById('data-table').classList.remove('hidden');
    document.getElementById('reupload-button').classList.remove('hidden');
    document.getElementById('upload-form').classList.add('hidden');
}

// Reset canvas
function resetCanvas() {
    document.getElementById('canvas-original').classList.add('hidden');
    document.getElementById('canvas-prediction').classList.add('hidden');
    document.getElementById('data-table').classList.add('hidden');
    document.getElementById('reupload-button').classList.add('hidden');
    document.getElementById('upload-form').classList.remove('hidden');
    document.getElementById('dropzone-file').value = '';
}

document.addEventListener('DOMContentLoaded', initialize);

// Initialize event listeners
function initialize() {
    setupDropzone();
    setupFileInput();
    setupReuploadButton();
}
