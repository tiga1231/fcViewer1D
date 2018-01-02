// assumes mathjs and numericjs

Array.prototype.t = function(x){
  return numeric.transpose(this);
};


Array.prototype.dot = function(y){
  return numeric.dot(this,y);
};


Array.prototype.sqrt = function(){
  return numeric.sqrt(this);
};

Array.prototype.pow = function(a){
  return numeric.pow(this,a);
};

Array.prototype.sum = function(){
  return numeric.sum(this);
};


Array.prototype.add = function(a){
  return numeric.add(this, a);
};


Array.prototype.sub = function(a){
  return numeric.sub(this, a);
};


Array.prototype.mul = function(a){
  return numeric.mul(this, a);
};


Array.prototype.div = function(a){
  return numeric.div(this, a);
};

Array.prototype.reshape = function(shape){
  return math.reshape(this, shape);
};

Array.prototype.max = function(){
  return math.max(this);
};

Array.prototype.min = function(){
  return math.min(this);
};

Array.prototype.abs = function(){
  return math.abs(this);
};


//linear algebra
var linalg = {
  eye: function(dim){
    return math.eye(dim)._data;
  },


  ones: function(d1, d2){
    return math.ones(d1, d2)._data;
  },


  zeros: function(d1, d2){
    return math.zeros(d1, d2)._data;
  },


  rand: function(d1, d2){
    var res = [];
    for (var i = 0; i < d1; i++) {
      var row = [];
      for (var j = 0; j < d2; j++) {
        row.push(Math.random());
      }
      res.push(row);
    }
    return res;
  },

  centeringMatrix: function(n){
    //centering matrix h
    var h = linalg.ones(n, n);
    h = numeric.mul(h, 1/n);
    h = numeric.sub(linalg.eye(n), h);
    return h;
  },

  distSquare: function(x){
    var n = x.length;
    var res = linalg.zeros(n,n);
    for (var i = 0; i < n; i++) {
      var x1 = x[i];
      for (var j = i+1; j < n; j++) {
        var x2 = x[j];
        var d2 = x1.sub(x2).pow(2).sum();
        res[i][j] = d2;
        res[j][i] = d2;
      }
    }
    return res;
  },

  kernelPCA: function(m, n_components=2, center=true){

    if(center=true){
      var n = m.length;
      //centering matrix h
      var h = linalg.centeringMatrix(n);
      //centering m by m' = hmh
      m = h.dot(m).dot(h);
    }

    var svd = numeric.svd(m);
    console.log(svd.S);
    var u2 = svd.U.t().slice(0,n_components).t();
    var s2 = numeric.diag(svd.S.slice(0,n_components));
    var x = u2.dot(s2.sqrt());
    return x;
  },


  MDS: function(d2, n_components=2){
    var n = d2.length;
    console.log(d2);
    var h = linalg.centeringMatrix(n);

    //centering m by m' = -1/2*hmh
    var m = h.dot(d2).dot(h).mul(-1/2);
    return linalg.kernelPCA(m, n_components=2, center=false);

  }


};

