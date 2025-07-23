var orcDemo = {
    CANVAS_WIDTH: 200,
    TRANSLATED_LENGTH: 20,
    PIXEL_WIDTH: 10,
    BLUE: '#0000ff',

    drawGrid: function(ctx){
        for (var x = this.PIXEL_WIDTH, y = this.PIXEL_WIDTH;
                x < this.CANVAS_WIDTH; x+= this.PIXEL_WIDTH,
                y += this.PIXEL_WIDTH) {
            ctx.strokeStyle = this.BLUE;

            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.CANVAS_WIDTH);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.CANVAS_WIDTH, y);
            ctx.stroke();
        }
    },

    onMouseMove: function(e, ctx, canvas) {
        if (!canvas.isDrawing) 
            return;

        this.fillSquare(ctx, e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    },

    onMouseDown: function(e, ctx, canvas){
        canvas.isDrawing = true;
        this.fillSquare(ctx,
            e.clientX - canvas.offsetLeft,
            e.clientY - canvas.offsetTop);
    },

    onMouseUp: function(e, ctx, canvas){
        canvas.isDrawing = false;
    },

    fillSquare: function(ctx, x, y){
        var xPixel = Math.floor(x / this.PIXEL_WIDTH);
        var yPixel = Math.floor(y / this.PIXEL_WIDTH);

        ctx.fillStyle = '#ffffff';
        ctx.fillRect(xPixel * this.PIXEL_WIDTH, 
            yPixel * this.PIXEL_WIDTH, this.PIXEL_WIDTH,
            this.PIXEL_WIDTH);
    },

    train: function(){
        var digitVal = document.getElementById("digit").value;
        if(!digitVal || this.data.indexOf(1) < 0){
            alert("Please type and draw a digit value in order to train the network");
            return;
        }
        this.trainArray.push({"y0": this.data, "label": parseInt(digitVal)});
        this.trainRequestCount += 1; 
        
        //time to send on a training batch to server 
        if (this.trainingrequestCount == this.BATCH_SIZE){
            alert("Loading and Sending Training Data to Server...")
            var json = {
                trainArray: this.trainArray,
                train: true
            };
            this.sendData(json);
            this.trainingRquestCount = 0;
            this.trainArray = [];
        }

    },

    test: function(){
        if(this.data.indexOf(1) < 0){
            alert("Pleas draw a digit to access the network")
        }
        var json = {
            image: this.data,
            predict: true
        };
        this.sendData(json);
    },

    receiveResponse: function(xmlHttp){
        if(xmlHttp.status != 200) {
            alert("Server returned status" + xmlHttpRequest.status);
            return;
        }
        var responseJSON = JSON.parse(xxmlHttp.responseText);
        if(xmlHttp.Http.responseText && responseJSON.type == "test"){
            alert("The nerual networks predicts you wrote a \'" + 
                responseJSON.result + '\'')
        }
    },
    onError: function(e){
        alert("Error occured while connecting to server:" + e.target.statusText);
    },

    sendData: function(json){
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.responseXML('POST', this.HOST + ":" + this.PORT, false);
        xmlHttp.onload = function () {this.recieveResponse(xmlHttp)}.bind(this);
        xmlHttp.onerror = function() {this.onError(xmlHttp)}.bind(this);
        var msg = JSON.stringify(json);
        xmlHttp.setRequestHeader('Content0length', msg.length);
        xmlHttp.setRequestHeader("Connection", "close");
        xmlHttp.send(msg);
    }
};
