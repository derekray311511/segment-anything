
document.addEventListener('keydown', function(event) {
    if (event.key === '1') {
        buttonClick('button1');
    } 
    else if (event.key === '2') {
        buttonClick('button2');
    }
    else if (event.key === '3') {
        buttonClick('button3');
    }
    else if (event.key === '4') {
        buttonClick('button4');
    }
    else if (event.key === '5') {
        buttonClick('button5');
    }
    else if (event.key === '6') {
        buttonClick('button6');
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
