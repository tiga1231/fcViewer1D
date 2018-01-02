var viz = {
  
  imshow: function(svg, img, sc=null, pixelSize=null){
    var width = svg.attr('width');
    var height = svg.attr('height');
    var short = Math.min(width, height);
    var margin = short * 0.05;

    pixelSize = pixelSize || Math.min( (height-2*margin)/img.length, (width-2*margin)/img[0].length );
    var data = viz.mat2obj(img);

    //TODO solve problem w/ img aspect ratio != canvas ratio
    var sx = d3.scaleLinear().domain([0, img[0].length]).range([margin, width-margin]);
    var sy = d3.scaleLinear().domain([0, img.length]).range([margin, height-margin]);

    var vmax = d3.max(data, d=>Math.abs(d.val));
    var vmin = -vmax;
    var sc = sc || d3.scaleLinear().domain([vmin, 0, vmax]).range(['#e66101','#f7f7f7', '#5e3c99']);

    svg.selectAll('.cell')
    .data(data)
    .enter()
    .append('rect')
    .attr('class', 'cell')
    .append('title');

    svg.selectAll('.cell')
    .data(data)
    .exit()
    .remove();

    svg.selectAll('.cell')
    .attr('x', d=> sx(d.j))
    .attr('y', d=> sy(d.i))
    .attr('width', pixelSize)
    .attr('height', pixelSize)
    .attr('fill', d=>sc(d.val));

    svg.selectAll('.cell')
    .selectAll('title')
    .text(d=>d.val);


  },


  _plot: function(svg, data, options){

    var sx = options.sx;
    var sy = options.sy;
    var sc = options.sc;

    var x = options.x;
    var y = options.y;

    svg.selectAll('x')
      .data(data)
      .enter()
      .append('circle')
      .attr('class', 'dot');

    svg.selectAll('.dot')
      .attr('cx', d=>sx(x(d)))
      .attr('cy', d=>sy(y(d)))
      .attr('r', 5)
      .attr('fill', d=>sc(d));

  },

  _xaxis: function(svg, options){
    svg.selectAll('.xaxis')
    .data([1])
    .enter()
    .append('g')
    .attr('class', 'xaxis');

    var tx = 0;
    var ty = options.height - options.margin;

    var ax = d3.axisBottom(options.sx);
    svg.select('.xaxis')
    .attr('transform', 'translate('+tx+','+ty+')')
    .call(ax);
  },

  _yaxis: function(svg, options){
    svg.selectAll('.yaxis')
    .data([1])
    .enter()
    .append('g')
    .attr('class', 'yaxis');

    var tx = options.margin;
    var ty = 0;

    var ay = d3.axisLeft(options.sy);
    svg.select('.yaxis')
    .attr('transform', 'translate('+tx+','+ty+')')
    .call(ay);
  },

  lines: function(svg, x){
    //x is a list of object:
    // {x1, y1, x2, y2, color}

    var width = svg.attr('width');
    var height = svg.attr('height');
    var short = Math.min(width, height);
    var margin = short * 0.1;

    var max = d3.max(x, d=>Math.max(Math.abs(d.x1), Math.abs(d.y1), Math.abs(d.x2), Math.abs(d.y2)) );
    var sx = svg.sx || d3.scaleLinear().domain([-max, max]).range([margin, width-margin]);
    var sy = svg.sy || d3.scaleLinear().domain([-max, max]).range([margin, height-margin]);
    console.log(max);

    svg.sx = sx;
    svg.sy = sy;

    svg.selectAll('x')
      .data(x)
      .enter()
      .append('line')
      .attr('class', 'line');

    svg.selectAll('.line')
      .attr('x1', d=>svg.sx(d.x1) )
      .attr('y1', d=>svg.sy(d.y1) )
      .attr('x2', d=>svg.sx(d.x2) )
      .attr('y2', d=>svg.sy(d.y2) )
      .attr('stroke', d=>d.color)
      .attr('stroke-width', d=>d.strokeWidth)
      .attr('opacity', d=>{
        if (d.opacity===undefined)
          return 1;
        else
          return d.opacity;
      });

  },


  plot: function(svg, x){

    var width = svg.attr('width');
    var height = svg.attr('height');
    var short = Math.min(width, height);
    var margin = short * 0.1;

    var getX = d=>d[0];
    var getY = d=>d[1];

    var max = d3.max(x, d=>Math.max(Math.abs(getX(d)), Math.abs(getY(d))) );
    var sx = svg.sx || d3.scaleLinear().domain([-max, max]).range([margin, width-margin]);
    var sy = svg.sy || d3.scaleLinear().domain([-max, max]).range([margin, height-margin]);

    svg.sx = sx;
    svg.sy = sy;

    var options = {
      width: width,
      height: height,
      margin: margin,
      x: getX,
      y: getY,
      sx: sx,
      sy: sy,
      sc: d=>'#5e3c99'
    };

    viz._plot(svg, x, options);
    viz._xaxis(svg, options);
    viz._yaxis(svg, options);
    
  },

  mat2obj: function(m){
    res = [];
    for (var i = 0; i < m.length; i++) {
      for (var j = 0; j < m[i].length; j++) {
        res.push({
          i:i,
          j:j,
          val:m[i][j]
        });
      }
    }
    return res;
  }

};
