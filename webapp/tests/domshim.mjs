/* Minimal DOM for rendering views under node.
 *
 * `disabled` models HTML semantics deliberately: a boolean attribute is true by
 * PRESENCE, so setAttribute('disabled','false') really does disable an element.
 * A shim that stored it as a plain string would let that bug through -- and it
 * did once, leaving every launcher control unusable while the tests passed.
 */
class N {
  constructor(t){this.tag=t;this.children=[];this.attrs={};this.style={};this.hidden=false;this.scrollTop=0;}
  setAttribute(k,v){this.attrs[k]=v;}
  getAttribute(k){return this.attrs[k];}
  hasAttribute(k){return k in this.attrs;}
  /** True exactly when the browser would treat the element as disabled. */
  get isDisabled(){return 'disabled' in this.attrs;}
  appendChild(c){this.children.push(c);return c;}
  replaceChildren(...c){this.children=c.filter(Boolean);}
  addEventListener(){}
  set className(v){this.attrs.class=v;} get className(){return this.attrs.class;}
  set textContent(v){this._t=v;} get textContent(){return this._t;}
  set innerHTML(v){this._h=v;} get innerHTML(){return this._h;}
  /** Depth-first search over the rendered tree. */
  find(pred){ if(pred(this)) return this;
    for(const c of this.children){ if(c && c.find){ const r=c.find(pred); if(r) return r; } }
    return null; }
  findAll(pred,out=[]){ if(pred(this)) out.push(this);
    for(const c of this.children){ if(c && c.findAll) c.findAll(pred,out); }
    return out; }
}
globalThis.document={createElement:(t)=>new N(t),createElementNS:(ns,t)=>new N(t),
  createTextNode:(t)=>({text:t}),getElementById:()=>new N('div')};
globalThis.window={addEventListener(){},__select(){}};
globalThis.location={hash:''};
globalThis.requestAnimationFrame=(f)=>f();
globalThis.fetch=()=>Promise.reject(new Error('no fetch in shim'));
globalThis.EventSource=class{constructor(){this.readyState=0;}addEventListener(){}close(){}};
