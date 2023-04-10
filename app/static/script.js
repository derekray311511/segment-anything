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
