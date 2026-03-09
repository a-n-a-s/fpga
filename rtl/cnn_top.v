module cnn_top #(
    parameter DATA_WIDTH    = 8,
    parameter ACC_WIDTH     = 32,
    parameter OUT_WIDTH     = 16,
    parameter CONV1_INPUT_LEN   = 12,
    parameter CONV1_NUM_FILTERS = 8,
    parameter CONV1_KERNEL_SIZE = 3,
    parameter CONV1_WEIGHT_ROM_DEPTH = 24,
    parameter POOL_SIZE         = 2,
    parameter POOL_STRIDE       = 2,
    parameter CONV2_NUM_FILTERS = 16,
    parameter CONV2_KERNEL_SIZE = 3,
    parameter CONV2_WEIGHT_ROM_DEPTH = 384,
    parameter FC_OUTPUT_SIZE    = 2,
    parameter FC_WEIGHT_ROM_DEPTH = 32,
    parameter DEBUG_DUMP      = 0  // Set to 1 to enable intermediate dumps
)(
    input  wire                             clk,
    input  wire                             rst,
    input  wire                             valid_in,
    input  wire signed [DATA_WIDTH-1:0]     data_in,
    output reg                              valid_out,
    output reg  [1:0]                       class_out,
    output reg  [7:0]                       confidence,       // Confidence score (0-255)
    output reg                              high_confidence,  // 1 if confidence > threshold
    output reg                              early_exit_taken, // 1 if early exit was taken
    output reg  [1:0]                       exit_layer      // Which layer exited at
);

    localparam CONV1_OUT_LEN = CONV1_INPUT_LEN;
    localparam POOL_OUT_LEN  = CONV1_OUT_LEN / 2;
    localparam CONV2_IN_CH   = CONV1_NUM_FILTERS;
    localparam CONV2_OUT_LEN = POOL_OUT_LEN;
    localparam GAP_NUM_FILTERS = CONV2_NUM_FILTERS;
    localparam FC_INPUT_SIZE = GAP_NUM_FILTERS;
    localparam signed [ACC_WIDTH-1:0] ACT_ZP = -128;  // Sign-extended to ACC_WIDTH
    localparam signed [ACC_WIDTH-1:0] OUT_ZP = -1;    // Output zero_point from 1:1 TFLite model
    localparam REQUANT_SHIFT = 20;
    localparam GAP_MULT = 32'd1722145;  // From 1:1 TFLite quantization

    localparam S_IDLE    = 4'd0;
    localparam S_LOAD    = 4'd1;
    localparam S_CONV1   = 4'd2;
    localparam S_POOL    = 4'd3;
    localparam S_EARLY_EXIT_CHECK = 4'd4;  // Check if we can exit early
    localparam S_CONV2   = 4'd5;
    localparam S_GAP     = 4'd6;
    localparam S_FC      = 4'd7;
    localparam S_RESULT  = 4'd8;  // Wait for logits to settle
    localparam S_ARGMAX  = 4'd9;
    localparam S_CONF    = 4'd10;  // Calculate confidence
    localparam S_DONE    = 4'd11;

    // Early exit layer codes
    localparam LAYER_EARLY_EXIT = 2'd0;  // Exited after Pool (Conv1+Pool)
    localparam LAYER_CONV2      = 2'd1;  // Exited after Conv2
    localparam LAYER_FULL       = 2'd2;  // Full network executed

    reg signed [DATA_WIDTH-1:0] input_buf [0:CONV1_INPUT_LEN-1];
    reg signed [DATA_WIDTH-1:0] conv1_buf [0:(CONV1_OUT_LEN*CONV1_NUM_FILTERS)-1];
    reg signed [DATA_WIDTH-1:0] pool_buf  [0:(POOL_OUT_LEN*CONV1_NUM_FILTERS)-1];
    reg signed [DATA_WIDTH-1:0] conv2_buf [0:(CONV2_OUT_LEN*CONV2_NUM_FILTERS)-1];
    reg signed [DATA_WIDTH-1:0] gap_buf   [0:GAP_NUM_FILTERS-1];

    reg [3:0] state;

    reg signed [DATA_WIDTH-1:0] conv1_weights_rom [0:CONV1_WEIGHT_ROM_DEPTH-1];
    reg signed [ACC_WIDTH-1:0] conv1_bias_rom [0:CONV1_NUM_FILTERS-1];
    reg signed [DATA_WIDTH-1:0] conv2_weights_rom [0:CONV2_WEIGHT_ROM_DEPTH-1];
    reg signed [ACC_WIDTH-1:0] conv2_bias_rom [0:CONV2_NUM_FILTERS-1];
    reg signed [DATA_WIDTH-1:0] dense_weights_rom [0:FC_WEIGHT_ROM_DEPTH-1];
    reg signed [ACC_WIDTH-1:0] dense_bias_rom [0:FC_OUTPUT_SIZE-1];

    reg [7:0] input_cnt;

    reg [3:0] conv_f;
    reg [7:0] conv_pos;
    reg [2:0] conv_k;

    reg [2:0] pool_f;
    reg [7:0] pool_pos;

    reg [3:0] conv2_in_ch;

    reg [3:0] gap_f;
    reg [7:0] gap_pos;

    reg [1:0] fc_o;
    reg [4:0] fc_i;

    reg signed [ACC_WIDTH-1:0] acc;
    reg signed [OUT_WIDTH-1:0] logit0;
    reg signed [OUT_WIDTH-1:0] logit1;
    reg [31:0] watchdog;

    // Confidence unit signals
    wire conf_done;
    reg  conf_start;
    wire [7:0] conf_score;
    wire conf_high_flag;

    // Early exit signals
    reg  early_exit_check;
    wire early_exit_decision;
    reg  [1:0] early_exit_class;  // Early exit prediction (based on Conv1 features)

    integer in_idx;

    function signed [DATA_WIDTH-1:0] sat_int8;
        input signed [ACC_WIDTH-1:0] x;
        begin
            if (x > 127)
                sat_int8 = 8'sd127;
            else if (x < -128)
                sat_int8 = -8'sd128;
            else
                sat_int8 = x[DATA_WIDTH-1:0];
        end
    endfunction

    function signed [OUT_WIDTH-1:0] sign_extend_8to16;
        input signed [7:0] x;
        begin
            sign_extend_8to16 = {{(OUT_WIDTH-8){x[7]}}, x};
        end
    endfunction

    function signed [OUT_WIDTH-1:0] sat_int16;
        input signed [ACC_WIDTH-1:0] x;
        begin
            if (x > 32767)
                sat_int16 = 16'sd32767;
            else if (x < -32768)
                sat_int16 = -16'sd32768;
            else
                sat_int16 = x[OUT_WIDTH-1:0];
        end
    endfunction

    function integer conv1_mult;
        input [3:0] ch;
        begin
            // All filters use the same multiplier: 6016 (from 1:1 TFLite quantization)
            conv1_mult = 6016;
        end
    endfunction

    function integer conv2_mult;
        input [3:0] ch;
        begin
            // All filters use the same multiplier: 5442 (from 1:1 TFLite quantization)
            conv2_mult = 5442;
        end
    endfunction

    function integer dense_mult;
        input [1:0] ch;
        begin
            // Both outputs use the same multiplier: 4399 (from 1:1 TFLite quantization)
            dense_mult = 4399;
        end
    endfunction

    function signed [DATA_WIDTH-1:0] requant_int8;
        input signed [ACC_WIDTH-1:0] x;
        input integer mult;
        input signed [DATA_WIDTH-1:0] zp;
        reg signed [63:0] prod;
        reg signed [63:0] scaled;
        begin
            prod = x * mult;
            if (prod >= 0)
                scaled = (prod + (1 <<< (REQUANT_SHIFT - 1))) >>> REQUANT_SHIFT;
            else
                scaled = (prod - (1 <<< (REQUANT_SHIFT - 1))) >>> REQUANT_SHIFT;
            scaled = scaled + zp;
            if (scaled > 127)
                requant_int8 = 8'sd127;
            else if (scaled < -128)
                requant_int8 = -8'sd128;
            else
                requant_int8 = scaled[DATA_WIDTH-1:0];
        end
    endfunction

    function signed [DATA_WIDTH-1:0] requant_relu_int8;
        input signed [ACC_WIDTH-1:0] x;
        input integer mult;
        reg signed [DATA_WIDTH-1:0] q;
        begin
            q = requant_int8(x, mult, ACT_ZP);
            if (q < ACT_ZP)
                requant_relu_int8 = ACT_ZP;
            else
                requant_relu_int8 = q;
        end
    endfunction

    initial begin
        $readmemh("data/conv1_weights_hex.mem", conv1_weights_rom);
        $readmemh("data/conv1_bias_hex.mem", conv1_bias_rom);
        $readmemh("data/conv2_weights_hex.mem", conv2_weights_rom);
        $readmemh("data/conv2_bias_hex.mem", conv2_bias_rom);
        $readmemh("data/dense_weights_hex.mem", dense_weights_rom);
        $readmemh("data/dense_bias_hex.mem", dense_bias_rom);
    end

    //========================================================================
    // Confidence Unit Instance
    //========================================================================
    confidence_unit #(
        .LOGIT_WIDTH(OUT_WIDTH),
        .CONF_WIDTH(8),
        .CONF_THRESHOLD(8'd180)  // 70% confidence threshold
    ) u_confidence (
        .clk(clk),
        .rst(rst),
        .start(conf_start),
        .logit0(logit0),
        .logit1(logit1),
        .confidence(conf_score),
        .high_confidence(conf_high_flag),
        .done(conf_done)
    );

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state      <= S_IDLE;
            valid_out  <= 1'b0;
            class_out  <= 2'd0;
            input_cnt  <= 8'd0;
            conv_f     <= 4'd0;
            conv_pos   <= 8'd0;
            conv_k     <= 3'd0;
            pool_f     <= 3'd0;
            pool_pos   <= 8'd0;
            conv2_in_ch<= 4'd0;
            gap_f      <= 4'd0;
            gap_pos    <= 8'd0;
            fc_o       <= 2'd0;
            fc_i       <= 5'd0;
            acc        <= {ACC_WIDTH{1'b0}};
            logit0     <= {OUT_WIDTH{1'b0}};
            logit1     <= {OUT_WIDTH{1'b0}};
            watchdog   <= 32'd0;
            conf_start <= 1'b0;
            confidence <= {8{1'b0}};
            high_confidence <= 1'b0;
            early_exit_taken <= 1'b0;
            exit_layer   <= 2'd0;
            early_exit_check <= 1'b0;
            early_exit_class <= 2'd0;
        end else begin
            valid_out <= 1'b0;
            if (state == S_IDLE)
                watchdog <= 32'd0;
            else
                watchdog <= watchdog + 32'd1;

            case (state)
                S_IDLE: begin
                    input_cnt <= 8'd0;
                    if (valid_in) begin
                        input_buf[0] <= data_in;
                        input_cnt    <= 8'd1;
                        state        <= S_LOAD;
                    end
                end

                S_LOAD: begin
                    if (valid_in) begin
                        input_buf[input_cnt] <= data_in;
                        if (input_cnt == CONV1_INPUT_LEN - 1) begin
                            conv_f   <= 4'd0;
                            conv_pos <= 8'd0;
                            conv_k   <= 3'd0;
                            acc      <= {ACC_WIDTH{1'b0}};
                            state    <= S_CONV1;
                        end else begin
                            input_cnt <= input_cnt + 8'd1;
                        end
                    end
                end

                // Conv1D: in_ch=1, out_ch=8, kernel=3, padding='same'
                S_CONV1: begin
`ifndef SYNTHESIS
                    if (conv_f == 0 && conv_pos == 0) begin
                        $display("DEBUG_CONV1_CYCLE: k=%0d, acc_before=%0d", conv_k, acc);
                    end
`endif
                    if (conv_k < CONV1_KERNEL_SIZE) begin
                        in_idx = $signed(conv_pos) + $signed(conv_k) - 1;
                            if ((in_idx >= 0) && (in_idx < CONV1_INPUT_LEN)) begin
                                acc <= acc +
                                   (({{(ACC_WIDTH-DATA_WIDTH){input_buf[in_idx][DATA_WIDTH-1]}}, input_buf[in_idx]} - ACT_ZP) *
                                    {{(ACC_WIDTH-DATA_WIDTH){conv1_weights_rom[(conv_f * CONV1_KERNEL_SIZE) + conv_k][DATA_WIDTH-1]}}, conv1_weights_rom[(conv_f * CONV1_KERNEL_SIZE) + conv_k]});
`ifndef SYNTHESIS
                                if (conv_f == 0 && conv_pos == 0) begin
                                    $display("  DEBUG_MAC: k=%0d, dezp=%0d * weight=%0d = %0d, acc_after=%0d",
                                        conv_k,
                                        {{(ACC_WIDTH-DATA_WIDTH){input_buf[in_idx][DATA_WIDTH-1]}}, input_buf[in_idx]} - ACT_ZP,
                                        $signed(conv1_weights_rom[(conv_f * CONV1_KERNEL_SIZE) + conv_k]),
                                        ({{(ACC_WIDTH-DATA_WIDTH){input_buf[in_idx][DATA_WIDTH-1]}}, input_buf[in_idx]} - ACT_ZP) *
                                        $signed(conv1_weights_rom[(conv_f * CONV1_KERNEL_SIZE) + conv_k]),
                                        acc + (({{(ACC_WIDTH-DATA_WIDTH){input_buf[in_idx][DATA_WIDTH-1]}}, input_buf[in_idx]} - ACT_ZP) *
                                        $signed(conv1_weights_rom[(conv_f * CONV1_KERNEL_SIZE) + conv_k])));
                                end
`endif
                            end
                        conv_k <= conv_k + 3'd1;
                    end else begin
                        conv1_buf[(conv_pos * CONV1_NUM_FILTERS) + conv_f] <=
                            requant_relu_int8(acc + conv1_bias_rom[conv_f], conv1_mult(conv_f));
`ifndef SYNTHESIS
                        if (conv_f == 0 && conv_pos == 0) begin
                            $display("DEBUG_CONV1_RESULT: acc=%0d, bias=%0d, acc+b=%0d, mult=%0d",
                                acc, conv1_bias_rom[conv_f], acc + conv1_bias_rom[conv_f], conv1_mult(conv_f));
                        end
`endif

                        acc    <= {ACC_WIDTH{1'b0}};
                        conv_k <= 3'd0;

                        if (conv_pos == CONV1_OUT_LEN - 1) begin
                            conv_pos <= 8'd0;
                            if (conv_f == CONV1_NUM_FILTERS - 1) begin
                                pool_f   <= 3'd0;
                                pool_pos <= 8'd0;
                                state    <= S_POOL;
                            end else begin
                                conv_f <= conv_f + 4'd1;
                            end
                        end else begin
                            conv_pos <= conv_pos + 8'd1;
                        end
                    end
                end

                S_POOL: begin
                    if (conv1_buf[(pool_pos * 2) * CONV1_NUM_FILTERS + pool_f] >=
                        conv1_buf[((pool_pos * 2) + 1) * CONV1_NUM_FILTERS + pool_f])
                        pool_buf[pool_pos * CONV1_NUM_FILTERS + pool_f] <= conv1_buf[(pool_pos * 2) * CONV1_NUM_FILTERS + pool_f];
                    else
                        pool_buf[pool_pos * CONV1_NUM_FILTERS + pool_f] <= conv1_buf[((pool_pos * 2) + 1) * CONV1_NUM_FILTERS + pool_f];

                    if (pool_pos == POOL_OUT_LEN - 1) begin
                        pool_pos <= 8'd0;
                        if (pool_f == CONV1_NUM_FILTERS - 1) begin
                            // Pool complete - check for early exit
                            state <= S_EARLY_EXIT_CHECK;
                        end else begin
                            pool_f <= pool_f + 3'd1;
                        end
                    end else begin
                        pool_pos <= pool_pos + 8'd1;
                    end
                end

                // Early Exit Check - decide whether to skip Conv2, GAP, FC
                // Simple heuristic: check if Conv1 features show clear class preference
                S_EARLY_EXIT_CHECK: begin
                    // Simple early exit classifier: sum of Conv1 features
                    // If sum is strongly positive or negative, exit early with prediction
                    begin
                        integer f;
                        reg signed [15:0] feature_sum;
                        feature_sum = 0;
                        for (f = 0; f < CONV1_NUM_FILTERS; f = f + 1) begin
                            feature_sum = feature_sum + conv1_buf[f];  // First position features
                        end
                        
                        // Early exit decision based on feature sum threshold
                        if (feature_sum > 16'd500) begin
                            // Strong Class 0 signal - exit early
                            early_exit_taken <= 1'b1;
                            exit_layer <= LAYER_EARLY_EXIT;
                            class_out <= 2'd0;  // Predict Class 0
                            confidence <= 8'd200;  // High confidence estimate
                            high_confidence <= 1'b1;
                            valid_out <= 1'b1;
`ifndef SYNTHESIS
                            $display("DEBUG_EARLY_EXIT_CLASS0: feature_sum=%0d, exiting after Pool", feature_sum);
`endif
                            state <= S_DONE;
                        end
                        else if (feature_sum < -16'd500) begin
                            // Strong Class 1 signal - exit early
                            early_exit_taken <= 1'b1;
                            exit_layer <= LAYER_EARLY_EXIT;
                            class_out <= 2'd1;  // Predict Class 1
                            confidence <= 8'd200;  // High confidence estimate
                            high_confidence <= 1'b1;
                            valid_out <= 1'b1;
`ifndef SYNTHESIS
                            $display("DEBUG_EARLY_EXIT_CLASS1: feature_sum=%0d, exiting after Pool", feature_sum);
`endif
                            state <= S_DONE;
                        end
                        else begin
                            // Uncertain - continue to full network
                            early_exit_taken <= 1'b0;
                            exit_layer <= LAYER_FULL;
                            conv_f      <= 4'd0;
                            conv_pos    <= 8'd0;
                            conv_k      <= 3'd0;
                            conv2_in_ch <= 4'd0;
                            acc         <= {ACC_WIDTH{1'b0}};
                            state       <= S_CONV2;
                        end
                    end
                end

                // Conv2D: in_ch=8, out_ch=16, kernel=3, padding='same'
                S_CONV2: begin
                    if (conv2_in_ch < CONV2_IN_CH) begin
                        if (conv_k < CONV2_KERNEL_SIZE) begin
                            in_idx = $signed(conv_pos) + $signed(conv_k) - 1;
                            if ((in_idx >= 0) && (in_idx < POOL_OUT_LEN)) begin
                                acc <= acc +
                                       (({{(ACC_WIDTH-DATA_WIDTH){pool_buf[in_idx * CONV1_NUM_FILTERS + conv2_in_ch][DATA_WIDTH-1]}}, pool_buf[in_idx * CONV1_NUM_FILTERS + conv2_in_ch]} - ACT_ZP) *
                                        {{(ACC_WIDTH-DATA_WIDTH){conv2_weights_rom[(((conv_f * CONV2_KERNEL_SIZE) + conv_k) * CONV2_IN_CH) + conv2_in_ch][DATA_WIDTH-1]}}, conv2_weights_rom[(((conv_f * CONV2_KERNEL_SIZE) + conv_k) * CONV2_IN_CH) + conv2_in_ch]});
                            end
                            conv_k <= conv_k + 3'd1;
                        end else begin
                            conv_k      <= 3'd0;
                            conv2_in_ch <= conv2_in_ch + 4'd1;
                        end
                    end else begin
                        conv2_buf[(conv_pos * CONV2_NUM_FILTERS) + conv_f] <=
                            requant_relu_int8(acc + conv2_bias_rom[conv_f], conv2_mult(conv_f));

                        acc         <= {ACC_WIDTH{1'b0}};
                        conv_k      <= 3'd0;
                        conv2_in_ch <= 4'd0;

                        if (conv_pos == CONV2_OUT_LEN - 1) begin
                            conv_pos <= 8'd0;
                            if (conv_f == CONV2_NUM_FILTERS - 1) begin
                                gap_f   <= 4'd0;
                                gap_pos <= 8'd0;
                                acc     <= {ACC_WIDTH{1'b0}};
                                state   <= S_GAP;
                            end else begin
                                conv_f <= conv_f + 4'd1;
                            end
                        end else begin
                            conv_pos <= conv_pos + 8'd1;
                        end
                    end
                end

                S_GAP: begin
                    if (gap_pos < CONV2_OUT_LEN) begin
                        acc     <= acc + {{(ACC_WIDTH-DATA_WIDTH){conv2_buf[gap_pos * CONV2_NUM_FILTERS + gap_f][DATA_WIDTH-1]}}, conv2_buf[gap_pos * CONV2_NUM_FILTERS + gap_f]} - ACT_ZP;
                        gap_pos <= gap_pos + 8'd1;
                    end else begin
                        // Add rounding: (acc + 3) / 6 instead of acc / 6
                        gap_buf[gap_f] <= requant_int8((acc + 3) / CONV2_OUT_LEN, GAP_MULT, ACT_ZP);
                        acc     <= {ACC_WIDTH{1'b0}};
                        gap_pos <= 8'd0;

                        if (gap_f == GAP_NUM_FILTERS - 1) begin
                            fc_o  <= 2'd0;
                            fc_i  <= 5'd0;
                            acc   <= {ACC_WIDTH{1'b0}};
                            state <= S_FC;
                        end else begin
                            gap_f <= gap_f + 4'd1;
                        end
                    end
                end

                S_FC: begin
                    if (fc_i < FC_INPUT_SIZE) begin
                        acc <= acc +
                               (({{(ACC_WIDTH-DATA_WIDTH){gap_buf[fc_i][DATA_WIDTH-1]}}, gap_buf[fc_i]} - ACT_ZP) *
                                {{(ACC_WIDTH-DATA_WIDTH){dense_weights_rom[(fc_o * FC_INPUT_SIZE) + fc_i][DATA_WIDTH-1]}}, dense_weights_rom[(fc_o * FC_INPUT_SIZE) + fc_i]});
                        fc_i <= fc_i + 5'd1;
                    end else begin
                        if (fc_o == 0) begin
                            logit0 <= sign_extend_8to16(requant_int8(acc + dense_bias_rom[fc_o], dense_mult(fc_o), OUT_ZP));
`ifndef SYNTHESIS
                            $display("DEBUG_FC0: acc=%0d, bias=%0d, logit0=%0d", acc, dense_bias_rom[fc_o], logit0);
`endif
                        end
                        else begin
                            logit1 <= sign_extend_8to16(requant_int8(acc + dense_bias_rom[fc_o], dense_mult(fc_o), OUT_ZP));
`ifndef SYNTHESIS
                            $display("DEBUG_FC1: acc=%0d, bias=%0d, logit1=%0d", acc, dense_bias_rom[fc_o], logit1);
`endif
                        end

                        acc  <= {ACC_WIDTH{1'b0}};
                        fc_i <= 5'd0;

                        if (fc_o == FC_OUTPUT_SIZE - 1) begin
                            state <= S_RESULT;  // Wait for logits to settle
                        end else begin
                            fc_o <= fc_o + 2'd1;
                        end
                    end
                end

                S_RESULT: begin
                    // Wait one cycle for logits to settle (non-blocking assignment delay)
                    state <= S_ARGMAX;
                end

                S_ARGMAX: begin
                    if ($signed(logit1) > $signed(logit0)) begin
                        class_out = 2'd1;  // Blocking assignment for immediate effect
                        $display("DEBUG_COMPARE: logit1(%0d) > logit0(%0d) = TRUE", logit1, logit0);
                    end
                    else begin
                        class_out = 2'd0;
                        $display("DEBUG_COMPARE: logit1(%0d) > logit0(%0d) = FALSE", logit1, logit0);
                    end
                    // Don't assert valid_out yet - wait for confidence
                    // valid_out <= 1'b1;

                    // Start confidence calculation
                    conf_start <= 1'b1;

                    // Debug display
`ifndef SYNTHESIS
                    $display("DEBUG_ARGMAX_PREVIEW: logit0=%0d, logit1=%0d, class=%0d", logit0, logit1, class_out);

                    // Dump intermediate tensors for parity checking
                    if (DEBUG_DUMP) begin
                        integer dump_fd;
                        integer idx;
                        dump_fd = $fopen("rtl_intermediates.dump", "w");

                        // Dump Conv1 output
                        $fwrite(dump_fd, "=== conv1 ===\n");
                        for (idx = 0; idx < CONV1_OUT_LEN * CONV1_NUM_FILTERS; idx = idx + 1) begin
                            $fwrite(dump_fd, "%0d\n", conv1_buf[idx]);
                        end

                        // Dump Pool output
                        $fwrite(dump_fd, "=== pool ===\n");
                        for (idx = 0; idx < POOL_OUT_LEN * CONV1_NUM_FILTERS; idx = idx + 1) begin
                            $fwrite(dump_fd, "%0d\n", pool_buf[idx]);
                        end

                        // Dump Conv2 output
                        $fwrite(dump_fd, "=== conv2 ===\n");
                        for (idx = 0; idx < CONV2_OUT_LEN * CONV2_NUM_FILTERS; idx = idx + 1) begin
                            $fwrite(dump_fd, "%0d\n", conv2_buf[idx]);
                        end

                        // Dump GAP output
                        $fwrite(dump_fd, "=== gap ===\n");
                        for (idx = 0; idx < GAP_NUM_FILTERS; idx = idx + 1) begin
                            $fwrite(dump_fd, "%0d\n", gap_buf[idx]);
                        end

                        // Dump FC logits
                        $fwrite(dump_fd, "=== fc ===\n");
                        $fwrite(dump_fd, "%0d\n", logit0);
                        $fwrite(dump_fd, "%0d\n", logit1);

                        $fclose(dump_fd);
                        $display("DEBUG: Dumped intermediates to rtl_intermediates.dump");
                    end
`endif
                    state <= S_CONF;  // Calculate confidence
                end

                S_CONF: begin
                    if (conf_done) begin
                        conf_start <= 1'b0;
                        // Capture confidence values for output
                        confidence <= conf_score;
                        high_confidence <= conf_high_flag;
                        // Set early exit flags (full network executed)
                        early_exit_taken <= 1'b0;
                        exit_layer <= LAYER_FULL;  // 2 = full network
`ifndef SYNTHESIS
                        $display("DEBUG_ARGMAX: logit0=%0d, logit1=%0d, class=%0d", logit0, logit1, class_out);
                        $display("DEBUG_CONF: confidence=%0d (%0d%%), high_confidence=%b",
                            conf_score, (conf_score*100)/255, conf_high_flag);
                        $display("DEBUG_EARLY_EXIT: taken=%b, layer=%0d (0=early, 1=conv2, 2=full)", early_exit_taken, exit_layer);
`endif
                        // Now assert valid_out - all results ready
                        valid_out <= 1'b1;
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    valid_out <= 1'b1;
                    if (!valid_in)
                        state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase

`ifndef SYNTHESIS
            if (watchdog > 32'd200000) begin
                $display("ERROR: cnn_top watchdog timeout at time %0t, state=%0d", $time, state);
                $stop;
            end
            if ((state == S_CONV1) && (conv_pos >= CONV1_OUT_LEN)) begin
                $display("ERROR: conv1 position out of range: %0d", conv_pos);
                $stop;
            end
            if ((state == S_CONV2) && (conv_pos >= CONV2_OUT_LEN)) begin
                $display("ERROR: conv2 position out of range: %0d", conv_pos);
                $stop;
            end
            if ((state == S_FC) && (fc_i > FC_INPUT_SIZE)) begin
                $display("ERROR: FC input index out of range: %0d", fc_i);
                $stop;
            end
`endif
        end
    end

endmodule
