num_compute_steps = 0
num_memory_slots = 0

scheduler_url ="" //"http://localhost:5000"

schedule = null;
memory_tracker = undefined;

function setup() {
    nt = parseInt($('#elementName').val());
    ncp = parseInt($('#state').val());
    get_schedule(nt, ncp, function(data) {
        schedule = data;
        $("#RunButton").toggle();
    });
    add_compute_step(nt);
    add_memory_slot(ncp);
    reverse_counter = num_compute_steps;
    memory_tracker = new Array(ncp);
}
timer_delay = 150;
schedule_counter = 0;
function run() {
    action = schedule[schedule_counter];
    action = JSON.parse(action);
    type = action.type;
    switch(type) {
        case "ADVANCE":
            advance(action);
            setTimeout(run, timer_delay);
            break;
        case "TAKESHOT":
            takeshot(action);
            setTimeout(run, timer_delay);
            break;
        case "REVERSE": reverse(action);
            setTimeout(run, timer_delay);
            break;
        case "LASTFW": advance(action);
            setTimeout(run, timer_delay);
            break;
        case "RESTORE": restore(action);
            setTimeout(run, timer_delay);
            break;
        case "TERMINATE": terminate();
            break;
        default: console.log("Unknown");
            console.log(action);
        }
    schedule_counter += 1;
}

currently_executing = {'type': '', 'number': -1, 'previous': -1}
function set_doing(type, number) {
    set_state(currently_executing.type, currently_executing.number, currently_executing.previous);
    
    currently_executing.type=type;
    currently_executing.number = number;
    currently_executing.previous = get_state(type, number);
    //console.log(currently_executing)
    set_state(type, number, STATE_DOING);
}

function advance() {
    set_doing("forward", action.capo);
}


function reverse(action) {
    set_state("reverse", action.capo+1, STATE_DOING);
    set_state("reverse", action.capo+2, STATE_DONE);
}


function takeshot(action) {
 
    if (typeof memory_tracker[action.ckp] == 'undefined') {
        memory_tracker[action.ckp] = action.capo;        
    } else{
        set_state("forward", memory_tracker[action.ckp], STATE_UNDONE)
        memory_tracker[action.ckp] = action.capo;        
    }
    set_state("memory", action.ckp, MEM_USED);

    set_state("forward", action.capo, STATE_SAVED);
    currently_executing = {'type': '', 'number': -1, 'previous': -1}
}

function set_state(type, number, state) {
    console.log(type, number, state)
    element_name = "#" + type + number;
    element = $(element_name);
    if(type=="memory") {
        change_memory_state(element, state);
    } else{
        change_compute_state(element, state);
    }
}

function restore(action) {
    compute_counter = action.capo;
}

function terminate() {
    set_state("reverse", 1, STATE_DONE);
}

function get_schedule(nt, ncp, cb) {
    url = scheduler_url + "/static_schedule?nt=" + nt.toString() + "&ncp=" + ncp.toString();
    $.getJSON(url).done(cb).fail(function(data, textStatus, errorThrown){
        console.log("Unable to get schedule");
    });
}

function add_compute_step(num=1) {
    for(var i=0; i<num; i++) {
        add_forward_step();
        add_reverse_step();
        num_compute_steps = num_compute_steps + 1;
    }
}

function add_memory_slot(num=1) {
    for(i=0; i<num; i++) {
        el_id = "memory" + (num_memory_slots + 1);
        parent = $("#memory_container");
        element = "<div class='memory_slot_free' id='" + el_id + "'></div>";
        parent.append(element);
        num_memory_slots = num_memory_slots + 1;
    }
}

function assert(condition) {
    if(condition != true) {
        console.log("Assertion violated!");
        //throw "Assertion Error";
    }
}

function add_forward_step(num=1) {
    for(i=0; i<num; i++) {
        el_id = "forward" + (num_compute_steps + 1);
        parent = $("#forward_container");
        element = "<div class='compute_step_forward_undone' id='" + el_id + "'></div>";
        parent.append(element);
    }
}

function add_reverse_step(num=1) {
    for(i=0; i<num; i++) {
        el_id = "reverse" + (num_compute_steps + 1);
        parent = $("#reverse_container");
        element = "<div class='compute_step_reverse_undone' id='" + el_id + "'></div>";
        parent.append(element);
    }
}
STATE_UNDONE = 0
STATE_DOING = 1
STATE_DONE = 2
STATE_SAVED = 3

DIR_UNKNOWN = 0;
DIR_FORWARD = 1;
DIR_REVERSE = 2;

class_names = [['compute_step_forward_undone', 'compute_step_forward_doing',
                'compute_step_forward_done', 'compute_step_forward_saved'],
               ['compute_step_reverse_undone', 'compute_step_reverse_doing',
                'compute_step_reverse_done']];

function change_compute_state(element, state) {
    direction = get_direction(element);
    if(direction == 0) {
        return;
    }

    element.removeClass();
    class_to_be_added = class_names[direction-1][state];

    element.addClass(class_to_be_added);
}

function get_state(type, number) {
    element = $("#"+type+number);
    if(type=="memory") {
        if(element.hasClass("memory_slot_free")) {
            return MEM_FREE;
        } else{
            return MEM_USED;
        }
    } else{
        direction = (type=="forward")?DIR_FORWARD:DIR_REVERSE;
        if(direction==DIR_UNKNOWN)
            return -1;
        else {
            //console.log(type, number, element.attr('class'))
            return class_names[direction-1].indexOf(element.attr('class'));
        }
    }
}

function get_direction(element) {
    forward_classes = ['compute_step_forward_undone', 'compute_step_forward_doing',
                       'compute_step_forward_done', 'compute_step_forward_saved'];
    reverse_classes = ['compute_step_reverse_undone', 'compute_step_reverse_doing',
                       'compute_step_reverse_done'];
    for(i = 0; i<forward_classes.length; i++) {
        if(element.hasClass(forward_classes[i])) {
            return DIR_FORWARD;
        }
    }

    for(i = 0; i<reverse_classes.length; i++) {
        if(element.hasClass(reverse_classes[i])) {
            return DIR_REVERSE;
        }
    }
    return DIR_UNKNOWN;
}

MEM_FREE = 0;
MEM_USED = 1;
function change_memory_state(element, state) {
    assert(state == MEM_FREE || state == MEM_USED);
    if(state == MEM_USED) {
        element.removeClass('memory_slot_free');
        element.addClass('memory_slot_used');
    } else {
        element.removeClass('memory_slot_used');
        element.addClass('memory_slot_free');
    }
}
