var switch_view = 0;

document.addEventListener('keydown', function(event) {
    if (event.key === '1') {
        buttonClick('button1');
        switch_view = 0;
    } 
    else if (event.key === '2') {
        buttonClick('button2');
        switch_view = 1;
    }
    else if (event.key === 'v') {
        if (switch_view === 0) {
            buttonClick('button2');
        }
        else if (switch_view === 1) {
            buttonClick('button1');
        }
        switch_view ^= 1;
    }
    else if (event.key === '3' || event.key === "c") {
        buttonClick('button3');
    }
    else if (event.key === '4') {
        buttonClick('button4');
    }
    else if (event.key === '5') {
        buttonClick('button5');
    }
    else if (event.key === '6' || event.key === "b") {
        buttonClick('button6');
    }
    else if (event.key === '7' || event.key === "r") {
        buttonClick('button7');
    }
    else if (event.key === '8') {
        buttonClick('button8');
    }
    else if (event.key === 'ArrowLeft') {
        buttonClick('prev-image');
    }
    else if (event.key === 'ArrowRight') {
        buttonClick('next-image');
    }
});

function buttonClick(buttonId) {
    const button = document.getElementById(buttonId);
    button.click();
}

document.getElementById('button1').addEventListener('click', function() {
    console.log('Button 1 clicked');
});

document.getElementById('button2').addEventListener('click', function() {
    console.log('Button 2 clicked');
});

document.getElementById('button3').addEventListener('click', function() {
    console.log('Button 3 clicked');
});

document.getElementById('button4').addEventListener('click', function() {
    console.log('Button 4 clicked');
});

document.getElementById('button5').addEventListener('click', function() {
    console.log('Button 5 clicked');
});

document.getElementById('button6').addEventListener('click', function() {
    console.log('Button 6 clicked');
});

document.getElementById('button7').addEventListener('click', function() {
    console.log('Button 7 clicked');
});

document.getElementById('button8').addEventListener('click', function() {
    console.log('Button 8 clicked');
});

document.getElementById('prev-image').addEventListener('click', function() {
    console.log('Button prev-image clicked');
});

document.getElementById('next-image').addEventListener('click', function() {
    console.log('Button next-image clicked');
});

// Mouse wheel event

// For #Thumbnail-container
function handleMouseWheelScroll(e) {
    const thumbnailContainer = document.getElementById("thumbnail-container");
    e.preventDefault();
    // Scroll horizontally based on the wheelDeltaY value
    thumbnailContainer.scrollLeft += e.deltaY * 2;
}
document.getElementById("thumbnail-container").addEventListener("wheel", handleMouseWheelScroll);

// For preview zoom in / zoom out
function handleMouseWheel(e) {
    e.preventDefault();
    const scaleFactor = 0.1;
    const preview = document.getElementById("preview");
    const container = document.getElementById("image-container");

    // Calculate the new width and height
    let newWidth = preview.clientWidth + (e.deltaY < 0 ? 2 : -1) * scaleFactor * preview.clientWidth;
    let newHeight = preview.clientHeight + (e.deltaY < 0 ? 2 : -1) * scaleFactor * preview.clientHeight;

    if (newWidth < 100 || newHeight < 100) {
        newWidth = container.clientWidth;
        newHeight = container.clientHeight;
        return;
    }

    // Maintain the aspect ratio
    const aspectRatio = preview.naturalWidth / preview.naturalHeight;
    newHeight = newWidth / aspectRatio;

    // Get the computed style of the preview element and parent elements
    const previewStyle = getComputedStyle(preview);
    const maxWidth = previewStyle.getPropertyValue('max-width');
    const maxHeight = previewStyle.getPropertyValue('max-height');
    
    // Calculate the exact max-width value in pixels
    const maxWidthValue = parseInt(maxWidth, 10);
    const maxHeightValue = parseInt(maxHeight, 10);
    if (newHeight > maxHeightValue || newWidth > maxWidthValue) {
        console.log("Return due to large size")
        return;
    }

    // Update the preview size
    preview.style.width = `${newWidth}px`;
    preview.style.height = `${newHeight}px`;

    // Adjust the container scroll position to zoom in/out from the image center
    const centerX = (container.scrollWidth - container.clientWidth) / 2;
    const centerY = (container.scrollHeight - container.clientHeight) / 2;
    container.scrollLeft = centerX;
    container.scrollTop = centerY;
    updatePointsAndBoxes()
}

document.getElementById("preview").addEventListener("wheel", handleMouseWheel);

function getScalingFactor(originalWidth, originalHeight, currentWidth, currentHeight) {
    return {
        x: currentWidth / originalWidth,
        y: currentHeight / originalHeight
    };
}

function updatePointsAndBoxes() {
    const originalWidth = $('#preview').data('originalWidth');
    const originalHeight = $('#preview').data('originalHeight');
    const currentWidth = $('#preview').width();
    const currentHeight = $('#preview').height();

    const scalingFactor = getScalingFactor(originalWidth, originalHeight, currentWidth, currentHeight);

    points.forEach(point => {
        point.style.left = parseFloat(point.dataset.originalX) * scalingFactor.x - 4 + 'px';
        point.style.top = parseFloat(point.dataset.originalY) * scalingFactor.y - 4 + 'px';
    });

    boxes.forEach(box => {
        box.style.left = parseFloat(box.dataset.originalX1) * scalingFactor.x + 'px';
        box.style.top = parseFloat(box.dataset.originalY1) * scalingFactor.y + 'px';
        box.style.width = (parseFloat(box.dataset.originalX2) - parseFloat(box.dataset.originalX1)) * scalingFactor.x + 'px';
        box.style.height = (parseFloat(box.dataset.originalY2) - parseFloat(box.dataset.originalY1)) * scalingFactor.y + 'px';
    });
}
