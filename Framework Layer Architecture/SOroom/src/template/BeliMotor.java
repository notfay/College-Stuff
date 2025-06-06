package template;

public class BeliMotor extends BeliKendaraan {

	public BeliMotor(String nama) {
        super(nama);
    }
	
	@Override
	public void validasiDokumen() {
		System.out.println("Motor : " +  getNama());
	}

    @Override
    public void prosesPembayaran() {
        System.out.println("Pembayaran dilakukan secara tunai...");
    }

    @Override
    public void ngurusSTNKBPKB() {
        System.out.println("Mengurus STNK & BPKB motor, estimasi 5 hari...");
    }

    @Override
    public void kirimKendaraan() {
        System.out.println("Motor dikirim...");
    }
}
