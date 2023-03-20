LITERAL,DECOMPOSITION,TRUE = 0,1,2
class Node:
    
    node_id=1
    def __init__(self, elements=None, type=DECOMPOSITION):
        self.elements = elements
        self.id = Node.node_id
        self.type=type
        if self.type==LITERAL:
            self.literal = elements
        Node.node_id += 1
    
    def __repr__(self):
        return str(self.id)
    
    def is_decomposition(self):
        return self.type ==  DECOMPOSITION
    
    def is_literal(self):
        return self.type ==  LITERAL
    
    def is_true(self):
        return self.type ==  TRUE
    
    def clear_bits(self,clear_data=False):
        """Recursively clears bits.  For use when recursively navigating an
        SDD by marking bits (not for use with SddNode.as_list).
        Set clear_data to True to also clear data."""
        if self._bit is False: return
        self._bit = False
        if clear_data: self.data = None
        if self.is_decomposition():
            for p,s in self.elements:
                p.clear_bits(clear_data=clear_data)
                s.clear_bits(clear_data=clear_data)
    
    def positive_iter(self,first_call=True,clear_data=False):
        """post-order (children before parents) generator, skipping false SDD
        nodes"""

        if not hasattr(self, '_bit'):
            self._bit = False

        if self._bit: return
        self._bit = True

        if self.is_decomposition():
            for p,s in self.elements:
                for node in p.positive_iter(first_call=False): yield node
                for node in s.positive_iter(first_call=False): yield node
        yield self

        if first_call:
            self.clear_bits(clear_data=clear_data)
