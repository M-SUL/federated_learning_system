class N {
  constructor(t){this.tag=t;this.children=[];this.attrs={};this.style={};this.hidden=false;this.scrollTop=0;}
  setAttribute(k,v){this.attrs[k]=v;} getAttribute(k){return this.attrs[k];}
  appendChild(c){this.children.push(c);return c;}
  replaceChildren(...c){this.children=c.filter(Boolean);}
  addEventListener(){} set className(v){this.attrs.class=v;} get className(){return this.attrs.class;}
  set textContent(v){this._t=v;} get textContent(){return this._t;}
  set innerHTML(v){this._h=v;} get innerHTML(){return this._h;}
}
globalThis.document={createElement:(t)=>new N(t),createElementNS:(ns,t)=>new N(t),
  createTextNode:(t)=>({text:t}),getElementById:()=>new N('div')};
globalThis.window={addEventListener(){},__select(){}};
globalThis.location={hash:''};
globalThis.requestAnimationFrame=(f)=>f();
globalThis.fetch=()=>Promise.reject(new Error('no fetch in shim'));
globalThis.EventSource=class{constructor(){this.readyState=0;}addEventListener(){}close(){}};
