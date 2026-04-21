#include "node.h"

/**
 * Improve bit dispersion of a given hash value h.
 */
std::size_t mix_hash(std::size_t h) {
    return (h ^ (h << 16) ^ 89869747UL) * 3644798167UL;
}

/*
 * ----------
 *  Node
 * ----------
 */

/**
 * Add child to this node.
 * - Updates this.children;
 * - Updates this.hash;
 * - Increases the layer of this node to be at least above the child's layer.
 * @param child The new child of this node.
 */
void Node::add_child(Node* child, bool negative) {
    if (type != NodeType::Or && type != NodeType::And) {
        throw std::runtime_error("Can only add children to AND/OR nodes");
    }
    if (negative && type != NodeType::Or) {
        throw std::runtime_error("Negative edges are only supported on OR nodes");
    }

    children.push_back({child, negative});
    std::size_t child_hash = child->hash;
    if (negative)
        child_hash = mix_hash(child_hash ^ 0x517cc1b727220a95ULL);
    hash ^= mix_hash(child_hash);
    std::size_t layer_bound = child->layer + 1;
    if (layer_bound%2 == 0 && type == NodeType::And) {
        layer_bound++; // And nodes must be in odd layers
    } else if (layer_bound%2 == 1 && type == NodeType::Or) {
        layer_bound++; // Or nodes must be in even layers
    }
    layer = std::max(layer, layer_bound);
}

/**
 * Useful for printing.
 * @return The label of this node.
 */
std::string Node::get_label() const {
    std::string labelName;
    switch (type) {
        case NodeType::True: labelName = "T"; break;
        case NodeType::False: labelName = "F"; break;
        case NodeType::Or: negate ? labelName = "NO" : labelName = "O"; break;
        case NodeType::And: negate ? labelName = "NA" : labelName = "A"; break;
        case NodeType::Leaf: labelName = "L"; break;
        default: // should not happen. Indicates node was deleted?
            throw std::runtime_error("Invalid node type");
    }
    return labelName + std::to_string(layer) + "/" + std::to_string(ix);
}

/**
 * Create a dummy parent who is one layer above this node.
 * This is needed to create a chain of dummy nodes such
 * that each node only has children in the previous adjacent layer.
 * @return The dummy parent.
 */
Node* Node::dummy_parent() {
    Node* dummy = (layer % 2 == 0) ? Node::createAndNode() : Node::createOrNode();
    dummy->add_child(this);
    return dummy;
}


Node* Node::createLiteralNode(Lit lit) {
    int ix = lit.internal_val();
    return new Node{
            NodeType::Leaf,
            ix,
            {},
            0,
            mix_hash(ix),
            false
    };
}

Node* Node::createAndNode(bool negate) {
    std::size_t hash_seed = 13643702618494718795UL;
    if (negate)
        hash_seed = mix_hash(hash_seed ^ 0x9e3779b97f4a7c15ULL);

    return new Node{
            NodeType::And,
            -1,
            {},
            0,
            hash_seed,
            negate
    };
}

Node* Node::createOrNode(bool negate) {
    std::size_t hash_seed = 10911628454825363117UL;
    if (negate)
        hash_seed = mix_hash(hash_seed ^ 0x9e3779b97f4a7c15ULL);

    return new Node{
            NodeType::Or,
            -1,
            {},
            0,
            hash_seed,
            negate
    };
}

Node* Node::createTrueNode() {
    return new Node{
            NodeType::True,
            1,
            {},
            0,
            10398838469117805359UL,
            false
    };
}

Node* Node::createFalseNode() {
    return new Node{
            NodeType::False,
            0,
            {},
            0,
            2055047638380880996UL,
            false
    };
}

void Node::negate_constant() {
    if (is_true()) {
        type = NodeType::False;
        ix = 0;
        hash = 2055047638380880996UL;
    } else if (is_false()) {
        type = NodeType::True;
        ix = 1;
        hash = 10398838469117805359UL;
    } else {
        throw std::runtime_error("NodePtr.negate() is only supported for true/false nodes");
    }
}


bool compareNode(const Node& first_node, const Node& second_node) {
    return first_node.hash < second_node.hash;
}
